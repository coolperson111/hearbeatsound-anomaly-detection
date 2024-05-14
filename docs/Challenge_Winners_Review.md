# Review of all Physionet 2022 Challenge Winners for Outcome Detection (Abnormality)

1. CUED_Acoustics:
- Outcome - 1st (11144); Murmur - 2nd (0.776)
- Preprocessing: Complex
- Features: Complex
- Model: Segmentation based HSMM + CNN --> Too complex for us :)


2. prna:
- Outcome - 2nd (11403); Murmur - 20th (0.694)
- Preprocessing:
    - Downsampling to 1000Hz
    - order 2 butterworth filter (25-400Hz)
    - z normalization for each vector to have 0 mean
    - each recording was dividied into multiple con-secutive non-overlapping 3-second segments and 
    - each seg-ment was labeled using the murmur label of the recording and outcome label of the subject
    - MFCCs calculated for each 3s segment
- Features:
    - Time Domain Embedding vector (dont exactly understand what an embedding vector is)
    + Frequencey domain MFCC embedding vector
- Model: CNN, LCNN (Light CNN (?)) + one layer of LSTM

3. Melbourne Kangas
- Outcome - 3rd (11735); Murmur - 29th (0.632)
- Preprocessing:
    - 10s segments - Padding where required
    - MFCCs
- Too complex method - leave

4. CeZIS
- Outcome - 4th (11916); Murmur - 8th (0.756)
- too complex method - leave

5. CAU_UMN
- Outcome - 5th (11933); Murmur - 5th (0.767)
- Preprocessing:
    - Not mentioned
    - data augmentation such as cutout, cutmix and mixup
- Features:
    - STFT, mel spectogram, CQT (Constant Q Transform)

6. HCCL
- Outcome - 6th (11943), Murmur - 21st (0.69)
- Preprocessing:
    - 12.5s segments (without overlap)
- Features:
    - spectograms (224x224)
    - specaugment for augmentation
- Model:
    - vision transformer (ViT)
    - too complex to implement (no understanding)

7. Listen2YourHeart
- Outcome - 7th (11946); Murmur - 13th (0.737)
- Preprocessing:
    - 2000Hz Downsampling
- Model:
    - Self supervised learning, dont really understand

8. uke-cardio
- Outcome - 8th (11990); Murmur - 15th (0.735)
- Preprocessing:
    - cutmix 
    - mel spectogram --> z transformed
    - cropped into 224 long time windows (no overlap)
    - padded short sequences with zeros



