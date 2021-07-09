# Decoder_models_for_Image_Captioning
Inject-type and merge acrhitectures

## Procedure to Train Model:
Clone the Repository to preserve directory structure.
Run following python codes:
- Init-inject: `python generator_init_inject.py`
- Par-inject: `python generator_par_inject.py`
- Pre-inject: `python generator_pre_inject.py`
- Merge: `python generator_merge.py`


 Layer | Decoder | CIDEr  | Bleu-4 | Bleu-3 | Bleu-2 | Bleu-1 | ROUGE-L | METEOR | SPICE |
 ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |    
 1 Layer (Without Dense) | Init-inject | 0.6001 | 0.1869 | 0.2828 | 0.4273 | 0.6232 | 0.4545 | 0.1993 | 0.1280 |
 1 Layer (Without Dense) | Par-inject | **0.6210**  |**0.1957**  | **0.2971** | **0.4430**  |**0.6336**  | **0.4621** | **0.2008** | **0.1295** |
 1 Layer (Without Dense) | Pre-inject | 0.5821 | 0.1893 | 0.2849 | 0.4273 | 0.6235 | 0.4549 | 0.1961 | 0.1240 |
1 Layer (Without Dense) | Merge | 0.5102 | 0.1665 | 0.2547 | 0.3930 | 0.5923 | 0.4279 | 0.1842 | 0.1170 |
 1 Layer (With Dense) | Init-inject | 0.5131 | 0.1696 | 0.2584 | 0.3950 | 0.5934 | 0.4316 | 0.1868 | 0.1140 |
1 Layer (With Dense) | Par-inject | **0.5903** | **0.1932** | **0.2910** | **0.4348** | **0.6304** | **0.4553** | **0.1961** | **0.1251** |
 1 Layer (With Dense) | Pre-inject |0.5848 | 0.1907 | 0.2883 | 0.4331 | 0.6325 | 0.4540 | 0.1948 | 0.1232 |
 1 Layer (With Dense) | Merge | 0.4953 | 0.1581 | 0.2443 | 0.3823 | 0.5857 | 0.4226 | 0.1809 | 0.1077 |
 3 Layers (Without Dense) | Init-inject | **0.6524** | **0.2045** | **0.3038** | **0.4476** | **0.6379** | **0.4640** | **0.2067** | **0.1349** |
 3 Layers (Without Dense) | Par-inject | 0.5693 | 0.1898 | 0.2850 | 0.4247 | 0.6200 | 0.4517 | 0.1938 | 0.1221 |
 3 Layers (Without Dense) | Pre-inject | 0.5522 | 0.1868 | 0.2814 | 0.4213 | 0.6169 | 0.4496 | 0.1908 | 0.1193 |
 3 Layers (Without Dense) | Merge | 0.4988 | 0.1623 | 0.2517 | 0.3904 | 0.5898 | 0.4261 | 0.1833 | 0.1164 |
 3 Layers (With Dense) | Init-inject | 0.3830 | 0.1387 | 0.2221 | 0.3592 | 0.5618 | 0.4090 | 0.1637 | 0.0910 |
 3 Layers (With Dense) | Par-inject | **0.5815** | **0.1933** | **.2881** | **0.4290** | **0.6240** | **0.4538** | **0.1945** | **0.1218** |
 3 Layers (With Dense) | Pre-inject | 0.5609 | 0.1885 | 0.2825 | 0.4222 | 0.6177 | 0.4511 | 0.1910 | 0.1188 |
 3 Layers (With Dense) | Merge | 0.4882 | 0.1595 | 0.2463 | 0.3847 | 0.5903 | 0.4236 | 0.1802 | 0.1077 |
