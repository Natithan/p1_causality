#!/bin/bash
# script to sync checkpoints created in liir servers to VSC servers, for those to pick up the pace (and go faster) as soon as they can start.
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/dv1_arwen_b32g2a1_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dv_1_ca_b8g8a1_checkpunten &
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/dv4_gimli_b64g1a1_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dv_4_ca_b8g8a1_checkpunten &
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/dv3_gimli_b64g1a1_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dv_3_ca_b8g8a1_checkpunten &
## rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/vi1_frodo_b32g2a1_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/vi_1_sk_b16g4a1_checkpunten &
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/vi3_gimli_b64g1a1_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/vi_3_sk_b16g4a1_checkpunten

#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/vi_2_undel_gimli_b64g1a8_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/vi_2_undel_sk_b64g4a2_checkpunten
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/vi_2_undel_arwen_b32g1a16_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/vi_3_undel_sk_b64g4a2_checkpunten
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/no_prior_2_undel_gimli_b64g1a8_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/no_prior_undel_sk_b64g4a2_checkpunten
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/dep_priorundeleted_gimli_b64g1a8_checkpunten/  vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dep_prior_undel_sk_b64g4a2_checkpunten
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/PP_bcorrect_ckpts/v6/VQA_bert_base_6layer_6conect-v6/ vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/PP_bcorrect_ckpts/v6/VQA_bert_base_6layer_6conect-v6
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/PP_bcorrect_ckpts/vilbert/VQA_bert_base_6layer_6conect-vilbert/ vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/PP_bcorrect_ckpts/vilbert/VQA_bert_base_6layer_6conect-vilbert
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/PP_bcorrect_ckpts/v6/RetrievalFlickr30k_bert_base_6layer_6conect-v6/ vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/PP_bcorrect_ckpts/v6/RetrievalFlickr30k_bert_base_6layer_6conect-v6

#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dv8_sk_b64g4a2_checkpunten vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dv8_sk_b64g8a1_checkpunten
#rsync -arvP -e "ssh -i ~/.ssh/id_rsa_4096" /cw/working-arwen/nathan/dv8_gimli_b64g2a4_checkpunten/ vsc33642@login.hpc.kuleuven.be:/scratch/leuven/336/vsc33642/dv8_sk_b64g4a2_checkpunten