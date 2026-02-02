
import os
import torch
# from Loss import Loss
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from DataLoader import NoisyCleanSet
from stft_loss import MultiResolutionSTFTLoss
# from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import LightningModule
# from AudioFeatures import InputFeature
import augment
from collections import OrderedDict, defaultdict
import pandas as pd
import scipy
import numpy as np
# from torchmetrics import ScaleInvariantSignalNoiseRatio
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio,signal_distortion_ratio,scale_invariant_signal_noise_ratio,signal_noise_ratio
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality

from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
# from helperFunctions  import HelperFunctions
import pandas as pd
import numpy as np
# from asteroid.losses import pairwise_neg_sisdr, PITLossWrapper,PairwiseNegSDR
from model import CausalSuDORMRF
# from model_NC import SuDORMRF
from torch.utils.data.dataloader import default_collate
# from DataLoader import DynamicMixtureDataset, DynamicMixtureDataLoader_DevTest,DynamicMixtureDataLoader_Train,DynamicMixtureDataLoader_Dev,DynamicMixtureDataLoader_Test,DynamicMixtureDataLoader_Train_Libri,DynamicMixtureDataLoader_Dev_Libri
# from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio,signal_distortion_ratio,scale_invariant_signal_noise_ratio,signal_noise_ratio
# from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
# from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
import torch.nn.functional as F
import compute_metrics 
import time
from clarity.evaluator.haspi import haspi_v2
from clarity.utils.audiogram import Audiogram
import scipy.signal as signal
# import noisereduce as nr
# from noisereduce.torchgate import TorchGate as TG
class Lightning(LightningModule):
    def __init__(self, config, model_class=CausalSuDORMRF):
        super().__init__()
        # self.save_hyperparameters()

        self.config = config
        self.batch_size = config["batch_size"]
        self.learning_rate = config["lr"]
        self.test_times = []
        self.metric_buffer = []
        self.gflops = None
        self.params_m = None
        self.sample_rate = config["sample_rate"]
        if model_class == CausalSuDORMRF:
            self.model = model_class(
                in_audio_channels=config["in_audio_channels"],
                out_channels=config["out_channels"],
                in_channels=config["in_channels"],
                num_blocks=config["num_blocks"],
                upsampling_depth=config["upsampling_depth"],
                enc_kernel_size=config["enc_kernel_size"],
                enc_num_basis=config["enc_num_basis"],
                num_sources=config["num_sources"]
            )
        else:
            self.model = model_class(
                out_channels=config["out_channels"],
                in_channels=config["in_channels"],
                num_blocks=config["num_blocks"],
                upsampling_depth=config["upsampling_depth"],
                enc_kernel_size=config["enc_kernel_size"],
                enc_num_basis=config["enc_num_basis"],
                num_sources=config["num_sources"]
            )


        augments = []
        self.num_workers = config["num_workers"]
        self.segment = 4.0 #for test put none
        self.stride = 0.5
        self.stft_sc_factor = 0.1
        self.stft_mag_factor = 0.1
        self.length = int(self.segment * self.sample_rate)
        self.stride = int(self.stride * self.sample_rate)
        self.shift = 8000
        self.bandmask = 0.2
        self.shift_same = True

         # DEMUCS augmentations
        augments.append(augment.Shift(self.shift, self.shift_same))
        augments.append(augment.Remix())
        augments.append(augment.BandMask(self.bandmask, sample_rate=self.sample_rate))
        
        self.augment = nn.Sequential(*augments)
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=self.stft_sc_factor,
                                                  factor_mag=self.stft_mag_factor)


    def forward(self, x):
        return self.model(x)
    def on_fit_start(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Size in megabytes assuming FP32 weights (4 bytes per parameter)
        model_size_mb = total_params * 4 / (1024 ** 2)

        # self.log("model/total_parameters",total_params)
        # self.log("model/trainable_parameters",trainable_params)
        # self.log("model/size_MB_fp32",model_size_mb)


        print(f"Model size (FP32): {model_size_mb:.2f} MB, total parameters: {total_params}, trainable parameters: {trainable_params}")

    # ---------------------
    # TRAINING STEP
    # ---------------------

    def training_step(self, batch, batch_idx):
        noisy = batch['noisy']   # (B, C, T)
        clean = batch['clean']   # (B, C, T)
 
        # =============================================================
        # (1) Create "sources" for augmentation: [noise, clean]
        # =============================================================
        noise = noisy - clean
        sources = torch.stack([noise, clean], dim=0)  # (2, B, C, T)

        # =============================================================
        # (2) Apply DEMUCS augmentations (training only)
        # =============================================================
        sources = self.augment(sources)   # nn.Sequential of augment modules

        # Split back
        noise, clean = sources
        noisy = noise + clean  # new augmented mixture, (B, C, T)

        # =============================================================
        # (3) Per-example std normalization BEFORE model
        # =============================================================
        # std over time dim (-1), keep dims so broadcasting works
        eps = 1e-8
        noisy_std = noisy.std(dim=-1, keepdim=True)  # (B, C, 1)
        noisy_norm = noisy / (noisy_std + eps)

        # =============================================================
        # (4) Forward pass on normalized input
        # =============================================================
        estimate_norm = self(noisy_norm)              # (B, C, T)
        estimate = estimate_norm * (noisy_std + eps)  # de-normalize

        # =============================================================
        # (5) Loss: L1 + MRSTFT, as in the paper
        # =============================================================
        l1 = F.l1_loss(estimate, clean)
        sc_loss, mag_loss = self.mrstftloss(
            estimate.squeeze(1), clean.squeeze(1)
        )
        loss = l1 + sc_loss + mag_loss

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss



    # ---------------------
    # VALIDATION SETUP
    # ---------------------

    def validation_step(self, batch, batch_idx):
        noisy = batch['noisy']   # (B, C, T)
        clean = batch['clean']   # (B, C, T)

        # print(f"noisy shape is {noisy.shape}, clean shape is {clean.shape}")

        # =============================================================
        # (1) Std normalization BEFORE model (same as training)
        # =============================================================
        eps = 1e-8
        noisy_std = noisy.std(dim=-1, keepdim=True)   # (B, C, 1)
        noisy_norm = noisy / (noisy_std + eps)

        # =============================================================
        # (2) Forward + de-normalize
        # =============================================================
        estimate_norm = self(noisy_norm)
        estimate = estimate_norm * (noisy_std + eps)

        # =============================================================
        # (3) Loss: same as training (but no grad)
        # =============================================================
        l1 = F.l1_loss(estimate, clean)
        sc_loss, mag_loss = self.mrstftloss(
            estimate.squeeze(1), clean.squeeze(1)
        )
        loss = l1 + sc_loss + mag_loss

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def safe_pesq(self, estimate, clean,sr):
        try:
            return perceptual_evaluation_speech_quality(estimate, clean, sr, 'wb')
        except Exception:
            print("one except")
            return float('nan')
    def calibrate_to_spl(self, audio, target_db=65.0):
        """
        Scales audio to a target dB SPL. 
        Assumes a digital full-scale (0 dB FS) corresponds to 120 dB SPL.
        THis for the HASPI metrics
        """
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2)) + 1e-8
        # Calculate current dB (relative to 1.0)
        current_db = 20 * np.log10(rms)
        
        # We assume 0 dBFS = 120 dB SPL (standard convention for HASPI/HAIM)
        # So a signal with RMS of 1.0 is 120 dB SPL.
        current_spl = 120 + current_db
        
        # Find the gain needed to reach target_db
        gain = 10**((target_db - current_spl) / 20)
        return audio * gain


    def apply_nalr(self, audio, sr, hl_levels, hl_freqs, n_taps=80):
        # 1. Calculate PTA
        pta = (hl_levels[1] + hl_levels[2] + hl_levels[3]) / 3
        
        # 2. X factor
        x = 0.05 * (hl_levels[1] + hl_levels[2] + hl_levels[3])
        
        # 3. NAL-R Correction factors (Added 8000 for 16kHz stability)
        c_factors = {250: -17, 500: -8, 1000: 1, 2000: -1, 4000: -2, 6000: -2, 8000: -2}
        
        gains_db = []
        for i, f in enumerate(hl_freqs):
            c = c_factors.get(f, -2) # Default to -2 for high freqs
            g = x + 0.31 * hl_levels[i] + c
            gains_db.append(max(0, g))

        # 4. Design the FIR Filter
        norm_freqs = np.array(hl_freqs) / (sr / 2)
        
        # Fix the Scipy ValueError: ensure strictly increasing grid ending at 1.0
        if norm_freqs[-1] >= 1.0:
            freq_grid = np.concatenate(([0], norm_freqs))
            gain_grid = np.concatenate(([gains_db[0]], gains_db))
        else:
            freq_grid = np.concatenate(([0], norm_freqs, [1]))
            gain_grid = np.concatenate(([gains_db[0]], gains_db, [gains_db[-1]]))
        
        linear_gain = 10 ** (np.array(gain_grid) / 20)
        taps = signal.firwin2(n_taps + 1, freq_grid, linear_gain)
        
        return signal.filtfilt(taps, 1, audio)



    # def test_step(self,batch,batch_idx):
    #     """
    #     Lightning calls this inside the test loop with the data from the test dataloader
    #     passed in as `batch`.
    #     """
    #     clean = batch['clean'] 
    #     noisy = batch['noisy']

    #     # loss_levels = np.array([35,  35 , 35,  40 , 50,  60]) 
    #     # # loss_levels = np.array([0,  0 , 0,  0 , 0,  0]) 
    #     # freqs = np.array([250, 500, 1000, 2000, 4000, 6000])
    #     loss_levels = np.array([20, 20, 25, 35, 45, 50,50]) 
    #     freqs = np.array([250, 500, 1000, 2000, 4000, 6000,8000])
    #     hi_audiogram = Audiogram(levels=loss_levels, frequencies=freqs)



    #     ##for the sake of inferenec time alone, we cut 1 sec. But metrics in paper are reported on the full audio length.
    #     sr = 16000

    #     #---for 1 sec audio inf to check inference time.

    #     # clean = clean[:,:,:sr]
    #     # noisy = noisy[:,:,:sr]

    #     # =============================================================
    #     # (1) Std normalization BEFORE model (same as training)
    #     # =============================================================
    #     eps = 1e-8
    #     noisy_std = noisy.std(dim=-1, keepdim=True)   # (B, C, 1)
    #     noisy_norm = noisy / (noisy_std + eps)
    #     # Warm-up only on first batch
    #     if batch_idx == 0:
    #         for _ in range(10):
    #             _ = self(noisy_norm)

    #     if self.device.type == "cuda":
    #         torch.cuda.synchronize()

    #     # =============================================================
    #     # (2) Forward + de-normalize
    #     # =============================================================
    #     start_time = time.perf_counter()

    #     estimate_norm = self(noisy_norm)

    #     if self.device.type == "cuda":
    #         torch.cuda.synchronize()

    #     elapsed_time = time.perf_counter() - start_time
    #     self.test_times.append(elapsed_time)


    #     estimate = estimate_norm * (noisy_std + eps)

    #     est = estimate[0,0,:].cpu().numpy()
    #     clean_num = clean[0,0,:].cpu().numpy()
    #     noisy_num = noisy[0,0,:].cpu().numpy()


    #     fitted_clean = self.apply_nalr(clean_num, sr, loss_levels, freqs)
    #     fitted_est   = self.apply_nalr(est, sr, loss_levels, freqs)
    #     fitted_noisy = self.apply_nalr(noisy_num, sr, loss_levels, freqs)
    #     # --- SAFETY ADDITION: PEAK NORMALIZE TO PREVENT CLIPPING, cause HASPI penalizes clipping a lot ---
    #     def safe_norm(x):
    #         peak = np.max(np.abs(x))
    #         if peak > 0:
    #             return x / (peak + 1e-8) * 0.9  # Keep peaks at -1dB
    #         return x

    #     fitted_clean = safe_norm(fitted_clean)
    #     fitted_est = safe_norm(fitted_est)
    #     fitted_noisy = safe_norm(fitted_noisy)

    #     #  # =============================================================
    #     # # (3) PESQ
    #     # # =============================================================    
    #     # pesq = self.safe_pesq(estimate, clean, sr)
    #     # pesq_un = self.safe_pesq(noisy, clean, sr)


    #     # #  # =============================================================
    #     # # # (4) STOI
    #     # # # =============================================================  

    #     # stoi = short_time_objective_intelligibility(estimate[0,0,:], clean[0,0,:], sr)
    #     # stoi_un = short_time_objective_intelligibility(noisy[0,0,:], clean[0,0,:], sr)
    #     # #  # =============================================================
    #     # # # (5) CSIG
    #     # # # =============================================================  


    #     # csig = compute_metrics.compute_csig(est, clean_num, sr) #csig = compute_metrics.compute_csig(estimate[0,0,:].cpu().numpy(), clean[0,0,:].numpy(), sr)
    #     # csig_un = compute_metrics.compute_csig(noisy_num, clean_num, sr)
    #     # # #  # =============================================================
    #     # # # # (6) CBAK
    #     # # # # ============================================================= 

    #     # cbak = compute_metrics.compute_cbak(est, clean_num, sr)
    #     # cbak_un = compute_metrics.compute_cbak(noisy_num, clean_num, sr)

    #     # # #  # =============================================================
    #     # # # # (7) COVL
    #     # # # # =============================================================  

    #     # cvol = compute_metrics.compute_covl(est, clean_num, sr)
    #     # cvol_un = compute_metrics.compute_covl(noisy_num, clean_num, sr)
    #     # #  # =============================================================
    #     # # # (8) HASPI for mild hearing loss
    #     # # # =============================================================   
    #     hi_haspi, _ = haspi_v2(
    #         reference=self.calibrate_to_spl(clean_num),
    #         reference_sample_rate=sr,
    #         processed=self.calibrate_to_spl(fitted_est),
    #         processed_sample_rate=sr,
    #         audiogram=hi_audiogram
    #     )

    #     hi_haspi_un, _ = haspi_v2(
    #         reference=self.calibrate_to_spl(clean_num),
    #         reference_sample_rate=sr,
    #         processed=self.calibrate_to_spl(fitted_noisy),
    #         processed_sample_rate=sr,
    #         audiogram=hi_audiogram
    #     )
    #     hi_haspi_perf,_ = haspi_v2(
    #         reference=self.calibrate_to_spl(clean_num),
    #         reference_sample_rate=sr,
    #         processed=self.calibrate_to_spl(fitted_clean),
    #         processed_sample_rate=sr,
    #         audiogram=hi_audiogram
    #     )
    #     # # print(f"stoi {stoi}, pesq :{pesq}, csig: {csig}, cbak: {cbak}, cvol: {cvol}")
    #     # print(f"stoi {stoi.item()}, pesq :{pesq.item()}, stoi unprocessed: {stoi_un.item()}, pesq_un: {pesq_un.item()},haspi_un: {hi_haspi_un}, haspi:{hi_haspi}, haspi_perf:{hi_haspi_perf} ")
    #     print(f"haspi_un: {hi_haspi_un}, haspi:{hi_haspi}, haspi_perf:{hi_haspi_perf} ")

    #     self.metric_buffer.append({
    #         # "stoi":stoi.item(), #sidr_p,
    #         # "pesq": pesq.item(),
    #         # "stoi_un" : stoi_un.item(),
    #         # "pesq_un" : pesq_un.item(),
    #         # "csig" : csig,
    #         # "csig_un" : csig_un,
    #         # "cbak": cbak,
    #         # "cbak_un": cbak_un,
    #         # "cvol": cvol,
    #         # "cvol_un":cvol_un,
    #         "haspi": hi_haspi,
    #         "hi_haspi_un":  hi_haspi_un,
    #         "inf_time": elapsed_time,
    #     })

        # if batch_idx ==80 or batch_idx ==400:

        #     torchaudio.save("noisy"+str(batch_idx)+".wav", noisy[0,:,:].cpu(), 16000)
        #     torchaudio.save("estimate"+str(batch_idx)+".wav", estimate[0,:,:].cpu(), 16000)
        #     torchaudio.save("clean"+str(batch_idx)+".wav", clean[0,:,:].cpu(), 16000)

    def test_step(self, batch, batch_idx):
        """
        just or HASPI
        """
        clean = batch['clean'] 
        noisy = batch['noisy']
        sr = 16000

        # 1. Define Standard N2/N3 (7 points for 16kHz)
        loss_levels = np.array([20, 20, 25, 35, 45, 50, 50]) 
        freqs = np.array([250, 500, 1000, 2000, 4000, 6000, 8000])
        hi_audiogram = Audiogram(levels=loss_levels, frequencies=freqs)

        # =============================================================
        # (1) Std normalization BEFORE model (same as training)
        # =============================================================
        eps = 1e-8
        noisy_std = noisy.std(dim=-1, keepdim=True)   # (B, C, 1)
        noisy_norm = noisy / (noisy_std + eps)
        # Warm-up only on first batch
        if batch_idx == 0:
            for _ in range(10):
                _ = self(noisy_norm)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # =============================================================
        # (2) Forward + de-normalize
        # =============================================================
        start_time = time.perf_counter()

        estimate_norm = self(noisy_norm)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_time = time.perf_counter() - start_time
        self.test_times.append(elapsed_time)


        estimate = estimate_norm * (noisy_std + eps)

        est = estimate[0,0,:].cpu().numpy()
        clean_num = clean[0,0,:].cpu().numpy()
        noisy_num = noisy[0,0,:].cpu().numpy()
 
        estimate = estimate_norm * (noisy_std + eps)
        est = estimate[0,0,:].cpu().numpy()
        clean_num = clean[0,0,:].cpu().numpy()
        noisy_num = noisy[0,0,:].cpu().numpy()

        # 2. Apply NAL-R
        fitted_clean = self.apply_nalr(clean_num, sr, loss_levels, freqs)
        fitted_est   = self.apply_nalr(est, sr, loss_levels, freqs)
        fitted_noisy = self.apply_nalr(noisy_num, sr, loss_levels, freqs)

        # 3. Standard HASPI Calibration: RMS Normalization
        # We normalize to 0.05 RMS. In HASPI, if level1=65, then 0.05 RMS results in 65dB SPL.
        def prep_for_haspi(x, target_rms=0.05):
            rms = np.sqrt(np.mean(x**2)) + 1e-9
            return x * (target_rms / rms)

        h_ref = prep_for_haspi(clean_num)      # Raw clean is the best reference
        h_est = prep_for_haspi(fitted_est)    # Enhanced + Fitted
        h_un  = prep_for_haspi(fitted_noisy)  # Noisy + Fitted
        h_perf = prep_for_haspi(fitted_clean) # Clean + Fitted

        # 4. HASPI Calculation
        # level1=65 is the default, which now correctly maps our 0.05 RMS to 65dB SPL.
        hi_haspi, _ = haspi_v2(h_ref, sr, h_est, sr, hi_audiogram, level1=65.0)
        hi_haspi_un, _ = haspi_v2(h_ref, sr, h_un, sr, hi_audiogram, level1=65.0)
        hi_haspi_perf, _ = haspi_v2(h_ref, sr, h_perf, sr, hi_audiogram, level1=65.0)

        print(f"haspi_un: {hi_haspi_un:.4f}, haspi: {hi_haspi:.4f}, haspi_perf: {hi_haspi_perf:.4f}")

        self.metric_buffer.append({
            "haspi": hi_haspi,
            "hi_haspi_un": hi_haspi_un,
            "inf_time": elapsed_time,
        })


    # def test_step(self, batch, batch_idx):
    #     """
    #     For timing alone
    #     """
    #     if batch_idx != 1:
    #         return
    #     sr =16000

    #     noisy = batch["noisy"]
    #     noisy = noisy[:,:,:sr]

    #     eps = 1e-8
    #     noisy_std = noisy.std(dim=-1, keepdim=True)
    #     noisy_norm = noisy / (noisy_std + eps)
    #     noisy_norm = noisy_norm.cpu()
    #     self.cpu()
    #     # warm-up
    #     for _ in range(10):
    #         _ = self(noisy_norm)
    #     # if self.device.type == "cuda":
    #     #     torch.cuda.synchronize()

    #     start = time.perf_counter()
    #     _ = self(noisy_norm)
    #     # if self.device.type == "cuda":
    #     #     torch.cuda.synchronize()
    #     elapsed = time.perf_counter() - start

    #     print(f"\nForward-pass wall-clock time: {elapsed:.6f} s")

    #     self.trainer.should_stop = True

    def on_test_end(self):
        df = pd.DataFrame(self.metric_buffer)
        save_path = "test_full_results_ckpt246_fp32_full_gpu_denoiser_metHaspi_N2.csv"
        df.to_csv(save_path)

        print("\n===== Average Test Metrics =====")
        print(df.mean())

        # pesq_values = torch.tensor([item["pesq"] for item in self.metric_buffer])
        # num_nans = torch.isnan(pesq_values).sum().item()


        # print("NaN PESQ (noisy baseline):", num_nans)
        # print("\nGFLOPs: {:.2f} G, Params: {:.2f} M".format(self.gflops, self.params_m))
        print("Avg Inference Time per Sample: {:.4f} sec".format(np.mean(self.test_times)))



    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     tensorboard_logs = {'val_loss': avg_loss}
    #     return {'val_loss': avg_loss, 'log': tensorboard_logs,
    #             'progress_bar': tensorboard_logs}

    # ---------------------
    # TRAINING SETUP
    # ---------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=3e-4,              # or self.config["lr"], but set that to 3e-4
            betas=(0.9, 0.999),
            weight_decay=0.0,
        )
        return optimizer

    def train_dataloader(self):
        tr_dataset = NoisyCleanSet("egs/val/tr/", length=self.length, stride=self.stride)
        #DynamicMixtureDataset
        return DataLoader(tr_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers,drop_last=True, pin_memory=True)#collate_fn=lambda b: self.dynamic_collate_fn(b, sample_rate=8000, segment_sec=4.0),
    def val_dataloader(self):
        val_dataset = NoisyCleanSet("egs/val/cv/", length=self.length, stride=self.stride)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        test_dataset = NoisyCleanSet("egs/val/tt/", length=None, stride=None, pad=False)
        return DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=self.num_workers, drop_last=True, pin_memory=True)
