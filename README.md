# Decoder_models_for_Image_Captioning
Inject-type and merge acrhitectures


Encoder  | Layer | Decoder | CIDEr  | Bleu-4 | Bleu-3 | Bleu-2 | Bleu-1 | ROUGE-L | METEOR | SPICE |
------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |    
Inception V3 | 1 Layer (Without Dense) | Init-inject | 0.6001 | 0.1869 | 0.2828 | 0.4273 | 0.6232 | 0.4545 | 0.1993 | 0.1280 |
Inception V3 | 1 Layer (Without Dense) | Par-inject | 0.6210  |0.1957  | 0.2971 | 0.4430  | 0.6336  | 0.4621 | 0.2008 | 0.1295 |
Inception V3 | 1 Layer (Without Dense) | Pre-inject | 0.5821 | 0.1893 | 0.2849 | 0.4273 | 0.6235 | 0.4549 | 0.1961 | 0.1240 |
Inception V3 | 1 Layer (Without Dense) | Merge | 0.5102 | 0.1665 | 0.2547 | 0.3930 | 0.5923 | 0.4279 | 0.1842 | 0.1170 |
