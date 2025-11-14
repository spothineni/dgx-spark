root@7d62de5c7fc1:/workspace/assets# python Llama3_8B_LoRA_finetuning.py
/usr/local/lib/python3.12/dist-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]

============================================================
LLAMA 3.1 8B LoRA FINE-TUNING CONFIGURATION
============================================================
Model: meta-llama/Llama-3.1-8B-Instruct
Batch size: 4
Sequence length: 2048
Number of epochs: 1
Learning rate: 0.0001
LoRA rank: 8
Dataset size: 500
============================================================

Loading model: meta-llama/Llama-3.1-8B-Instruct
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [01:42<00:00, 25.54s/it]
Trainable parameters = 20,971,520
Loading dataset with 500 samples...
The model is already on multiple devices. Skipping the move to device specified in `args`.
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'eos_token_id': 128009, 'pad_token_id': 128009}.
  0%|                                                                                                                                              | 0/2 [00:00<?, ?it/s]W1114 18:14:09.265000 149 torch/_inductor/utils.py:1545] [0/0] Not enough SMs to use max_autotune_gemm mode
{'loss': 2.4295, 'grad_norm': 2.5387463569641113, 'learning_rate': 0.0001, 'num_tokens': 327.0, 'mean_token_accuracy': 0.44272446632385254, 'epoch': 0.01}               
{'loss': 2.2494, 'grad_norm': 1.8955515623092651, 'learning_rate': 5e-05, 'num_tokens': 607.0, 'mean_token_accuracy': 0.5036231875419617, 'epoch': 0.02}                 
{'train_runtime': 138.9395, 'train_samples_per_second': 0.036, 'train_steps_per_second': 0.014, 'train_loss': 2.339435338973999, 'epoch': 0.02}                          
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [02:18<00:00, 69.47s/it]

Starting LoRA fine-tuning for 1 epoch(s)...
The model is already on multiple devices. Skipping the move to device specified in `args`.
{'loss': 2.0353, 'grad_norm': 1.78031325340271, 'learning_rate': 0.0001, 'num_tokens': 327.0, 'mean_token_accuracy': 0.5325077176094055, 'epoch': 0.01}                  
{'loss': 1.8322, 'grad_norm': 2.2500431537628174, 'learning_rate': 9.92e-05, 'num_tokens': 607.0, 'mean_token_accuracy': 0.5615941882133484, 'epoch': 0.02}              
{'loss': 1.7221, 'grad_norm': 1.8881832361221313, 'learning_rate': 9.84e-05, 'num_tokens': 972.0, 'mean_token_accuracy': 0.5318559408187866, 'epoch': 0.02}              
{'loss': 1.3563, 'grad_norm': 2.1766700744628906, 'learning_rate': 9.76e-05, 'num_tokens': 1310.0, 'mean_token_accuracy': 0.6616766452789307, 'epoch': 0.03}             
{'loss': 1.44, 'grad_norm': 2.400846242904663, 'learning_rate': 9.680000000000001e-05, 'num_tokens': 1736.0, 'mean_token_accuracy': 0.6184834241867065, 'epoch': 0.04}   
{'loss': 1.4412, 'grad_norm': 2.616469621658325, 'learning_rate': 9.6e-05, 'num_tokens': 2073.0, 'mean_token_accuracy': 0.6996996998786926, 'epoch': 0.05}               
{'loss': 1.3048, 'grad_norm': 3.992102861404419, 'learning_rate': 9.52e-05, 'num_tokens': 2475.0, 'mean_token_accuracy': 0.6809045076370239, 'epoch': 0.06}              
{'loss': 1.2156, 'grad_norm': 1.0286370515823364, 'learning_rate': 9.44e-05, 'num_tokens': 3194.0, 'mean_token_accuracy': 0.6909090876579285, 'epoch': 0.06}             
{'loss': 0.9996, 'grad_norm': 1.4361685514450073, 'learning_rate': 9.360000000000001e-05, 'num_tokens': 3596.0, 'mean_token_accuracy': 0.7487437129020691, 'epoch': 0.07}
{'loss': 0.818, 'grad_norm': 1.5021713972091675, 'learning_rate': 9.28e-05, 'num_tokens': 3896.0, 'mean_token_accuracy': 0.8006756901741028, 'epoch': 0.08}              
{'loss': 0.9624, 'grad_norm': 1.519971251487732, 'learning_rate': 9.200000000000001e-05, 'num_tokens': 4204.0, 'mean_token_accuracy': 0.7730262875556946, 'epoch': 0.09} 
{'loss': 0.7332, 'grad_norm': 1.986219882965088, 'learning_rate': 9.120000000000001e-05, 'num_tokens': 4457.0, 'mean_token_accuracy': 0.8353413939476013, 'epoch': 0.1}  
{'loss': 1.2525, 'grad_norm': 1.3263386487960815, 'learning_rate': 9.04e-05, 'num_tokens': 4954.0, 'mean_token_accuracy': 0.663286030292511, 'epoch': 0.1}               
{'loss': 0.8054, 'grad_norm': 1.6523126363754272, 'learning_rate': 8.960000000000001e-05, 'num_tokens': 5324.0, 'mean_token_accuracy': 0.7978141903877258, 'epoch': 0.11}
{'loss': 1.26, 'grad_norm': 1.4726828336715698, 'learning_rate': 8.88e-05, 'num_tokens': 5842.0, 'mean_token_accuracy': 0.6673151850700378, 'epoch': 0.12}               
{'loss': 0.8869, 'grad_norm': 2.017130136489868, 'learning_rate': 8.800000000000001e-05, 'num_tokens': 6152.0, 'mean_token_accuracy': 0.7745097875595093, 'epoch': 0.13} 
{'loss': 1.0482, 'grad_norm': 2.5013680458068848, 'learning_rate': 8.72e-05, 'num_tokens': 6451.0, 'mean_token_accuracy': 0.7796609997749329, 'epoch': 0.14}             
{'loss': 0.889, 'grad_norm': 2.0861282348632812, 'learning_rate': 8.64e-05, 'num_tokens': 6771.0, 'mean_token_accuracy': 0.7848101258277893, 'epoch': 0.14}              
{'loss': 1.0647, 'grad_norm': 1.5189757347106934, 'learning_rate': 8.560000000000001e-05, 'num_tokens': 7275.0, 'mean_token_accuracy': 0.7200000286102295, 'epoch': 0.15}
{'loss': 1.1886, 'grad_norm': 1.329514741897583, 'learning_rate': 8.48e-05, 'num_tokens': 7803.0, 'mean_token_accuracy': 0.7080152630805969, 'epoch': 0.16}              
{'loss': 1.1449, 'grad_norm': 1.8616290092468262, 'learning_rate': 8.4e-05, 'num_tokens': 8234.0, 'mean_token_accuracy': 0.7377049326896667, 'epoch': 0.17}              
{'loss': 1.2096, 'grad_norm': 1.364135503768921, 'learning_rate': 8.32e-05, 'num_tokens': 8763.0, 'mean_token_accuracy': 0.668571412563324, 'epoch': 0.18}               
{'loss': 1.1319, 'grad_norm': 1.4471665620803833, 'learning_rate': 8.24e-05, 'num_tokens': 9222.0, 'mean_token_accuracy': 0.7186813354492188, 'epoch': 0.18}             
{'loss': 1.0658, 'grad_norm': 1.4914463758468628, 'learning_rate': 8.16e-05, 'num_tokens': 9566.0, 'mean_token_accuracy': 0.7617647051811218, 'epoch': 0.19}             
{'loss': 0.9824, 'grad_norm': 1.1057238578796387, 'learning_rate': 8.080000000000001e-05, 'num_tokens': 9971.0, 'mean_token_accuracy': 0.7406483888626099, 'epoch': 0.2} 
{'loss': 1.1201, 'grad_norm': 1.0310641527175903, 'learning_rate': 8e-05, 'num_tokens': 10339.0, 'mean_token_accuracy': 0.75, 'epoch': 0.21}                             
{'loss': 1.1844, 'grad_norm': 0.8607496619224548, 'learning_rate': 7.920000000000001e-05, 'num_tokens': 10974.0, 'mean_token_accuracy': 0.6973058581352234, 'epoch': 0.22}
{'loss': 0.6949, 'grad_norm': 0.9334842562675476, 'learning_rate': 7.840000000000001e-05, 'num_tokens': 11290.0, 'mean_token_accuracy': 0.817307710647583, 'epoch': 0.22}
{'loss': 0.8589, 'grad_norm': 0.8479636311531067, 'learning_rate': 7.76e-05, 'num_tokens': 11719.0, 'mean_token_accuracy': 0.7717646956443787, 'epoch': 0.23}            
{'loss': 1.0503, 'grad_norm': 0.9159661531448364, 'learning_rate': 7.680000000000001e-05, 'num_tokens': 12160.0, 'mean_token_accuracy': 0.7482837438583374, 'epoch': 0.24}
{'loss': 0.9623, 'grad_norm': 0.8212323784828186, 'learning_rate': 7.6e-05, 'num_tokens': 12587.0, 'mean_token_accuracy': 0.7423167824745178, 'epoch': 0.25}             
{'loss': 1.1001, 'grad_norm': 0.7506073713302612, 'learning_rate': 7.52e-05, 'num_tokens': 13115.0, 'mean_token_accuracy': 0.7003816962242126, 'epoch': 0.26}            
{'loss': 0.9876, 'grad_norm': 0.8939354419708252, 'learning_rate': 7.44e-05, 'num_tokens': 13500.0, 'mean_token_accuracy': 0.748031497001648, 'epoch': 0.26}             
{'loss': 0.82, 'grad_norm': 0.8144983649253845, 'learning_rate': 7.36e-05, 'num_tokens': 13943.0, 'mean_token_accuracy': 0.8086560368537903, 'epoch': 0.27}              
{'loss': 0.6748, 'grad_norm': 0.7865234017372131, 'learning_rate': 7.280000000000001e-05, 'num_tokens': 14355.0, 'mean_token_accuracy': 0.8333333134651184, 'epoch': 0.28}
{'loss': 0.9771, 'grad_norm': 1.1628577709197998, 'learning_rate': 7.2e-05, 'num_tokens': 14649.0, 'mean_token_accuracy': 0.7724137902259827, 'epoch': 0.29}             
{'loss': 0.9804, 'grad_norm': 0.8327434659004211, 'learning_rate': 7.12e-05, 'num_tokens': 15066.0, 'mean_token_accuracy': 0.7530266046524048, 'epoch': 0.3}             
{'loss': 0.8888, 'grad_norm': 1.082124948501587, 'learning_rate': 7.04e-05, 'num_tokens': 15415.0, 'mean_token_accuracy': 0.7710145115852356, 'epoch': 0.3}              
{'loss': 1.0889, 'grad_norm': 0.8559914827346802, 'learning_rate': 6.96e-05, 'num_tokens': 15833.0, 'mean_token_accuracy': 0.7222222089767456, 'epoch': 0.31}            
{'loss': 0.8043, 'grad_norm': 0.9828931093215942, 'learning_rate': 6.879999999999999e-05, 'num_tokens': 16155.0, 'mean_token_accuracy': 0.7767295837402344, 'epoch': 0.32}
{'loss': 0.9557, 'grad_norm': 1.3558950424194336, 'learning_rate': 6.800000000000001e-05, 'num_tokens': 16499.0, 'mean_token_accuracy': 0.770588219165802, 'epoch': 0.33}
{'loss': 1.181, 'grad_norm': 0.7823352813720703, 'learning_rate': 6.720000000000001e-05, 'num_tokens': 16994.0, 'mean_token_accuracy': 0.6924643516540527, 'epoch': 0.34}
{'loss': 0.8224, 'grad_norm': 0.9150792956352234, 'learning_rate': 6.64e-05, 'num_tokens': 17294.0, 'mean_token_accuracy': 0.7939189076423645, 'epoch': 0.34}            
{'loss': 1.0492, 'grad_norm': 0.8286798596382141, 'learning_rate': 6.560000000000001e-05, 'num_tokens': 17680.0, 'mean_token_accuracy': 0.7146596908569336, 'epoch': 0.35}
{'loss': 1.2077, 'grad_norm': 0.7925069332122803, 'learning_rate': 6.48e-05, 'num_tokens': 18069.0, 'mean_token_accuracy': 0.6883116960525513, 'epoch': 0.36}            
{'loss': 0.8043, 'grad_norm': 0.883600115776062, 'learning_rate': 6.400000000000001e-05, 'num_tokens': 18487.0, 'mean_token_accuracy': 0.8212560415267944, 'epoch': 0.37}
{'loss': 1.1576, 'grad_norm': 0.6683416962623596, 'learning_rate': 6.32e-05, 'num_tokens': 19020.0, 'mean_token_accuracy': 0.6824196577072144, 'epoch': 0.38}            
{'loss': 0.8921, 'grad_norm': 0.8125153183937073, 'learning_rate': 6.24e-05, 'num_tokens': 19444.0, 'mean_token_accuracy': 0.7666666507720947, 'epoch': 0.38}            
{'loss': 1.0732, 'grad_norm': 0.9071339964866638, 'learning_rate': 6.16e-05, 'num_tokens': 19818.0, 'mean_token_accuracy': 0.7108108401298523, 'epoch': 0.39}            
{'loss': 0.9875, 'grad_norm': 0.895899772644043, 'learning_rate': 6.08e-05, 'num_tokens': 20166.0, 'mean_token_accuracy': 0.75, 'epoch': 0.4}                            
{'loss': 1.064, 'grad_norm': 0.6879331469535828, 'learning_rate': 6e-05, 'num_tokens': 20702.0, 'mean_token_accuracy': 0.6879699230194092, 'epoch': 0.41}                
{'loss': 1.0436, 'grad_norm': 1.0031907558441162, 'learning_rate': 5.92e-05, 'num_tokens': 21129.0, 'mean_token_accuracy': 0.7399527430534363, 'epoch': 0.42}            
{'loss': 1.1938, 'grad_norm': 0.6266855597496033, 'learning_rate': 5.8399999999999997e-05, 'num_tokens': 21892.0, 'mean_token_accuracy': 0.6903820633888245, 'epoch': 0.42}
{'loss': 1.0171, 'grad_norm': 0.9632011651992798, 'learning_rate': 5.76e-05, 'num_tokens': 22289.0, 'mean_token_accuracy': 0.7379134893417358, 'epoch': 0.43}            
{'loss': 0.6101, 'grad_norm': 1.013504147529602, 'learning_rate': 5.68e-05, 'num_tokens': 22577.0, 'mean_token_accuracy': 0.8239436745643616, 'epoch': 0.44}             
{'loss': 1.0526, 'grad_norm': 0.8901793956756592, 'learning_rate': 5.6000000000000006e-05, 'num_tokens': 23120.0, 'mean_token_accuracy': 0.7068645358085632, 'epoch': 0.45}
{'loss': 0.8853, 'grad_norm': 0.7964140772819519, 'learning_rate': 5.520000000000001e-05, 'num_tokens': 23545.0, 'mean_token_accuracy': 0.745843231678009, 'epoch': 0.46}
{'loss': 0.9107, 'grad_norm': 0.7743248343467712, 'learning_rate': 5.440000000000001e-05, 'num_tokens': 23965.0, 'mean_token_accuracy': 0.754807710647583, 'epoch': 0.46}
{'loss': 1.2693, 'grad_norm': 0.9043445587158203, 'learning_rate': 5.360000000000001e-05, 'num_tokens': 24458.0, 'mean_token_accuracy': 0.7075664401054382, 'epoch': 0.47}
{'loss': 0.8241, 'grad_norm': 0.883080005645752, 'learning_rate': 5.28e-05, 'num_tokens': 24777.0, 'mean_token_accuracy': 0.7904762029647827, 'epoch': 0.48}             
{'loss': 1.1054, 'grad_norm': 0.7669453024864197, 'learning_rate': 5.2000000000000004e-05, 'num_tokens': 25227.0, 'mean_token_accuracy': 0.7174887657165527, 'epoch': 0.49}
{'loss': 0.9198, 'grad_norm': 0.6035386919975281, 'learning_rate': 5.1200000000000004e-05, 'num_tokens': 25821.0, 'mean_token_accuracy': 0.7559322118759155, 'epoch': 0.5}
{'loss': 1.0207, 'grad_norm': 0.6694092750549316, 'learning_rate': 5.0400000000000005e-05, 'num_tokens': 26336.0, 'mean_token_accuracy': 0.7260273694992065, 'epoch': 0.5}
{'loss': 0.61, 'grad_norm': 1.5063974857330322, 'learning_rate': 4.96e-05, 'num_tokens': 26601.0, 'mean_token_accuracy': 0.8275862336158752, 'epoch': 0.51}              
{'loss': 0.7738, 'grad_norm': 0.7918925881385803, 'learning_rate': 4.88e-05, 'num_tokens': 26994.0, 'mean_token_accuracy': 0.7789202928543091, 'epoch': 0.52}            
{'loss': 0.947, 'grad_norm': 0.7602059245109558, 'learning_rate': 4.8e-05, 'num_tokens': 27418.0, 'mean_token_accuracy': 0.7404761910438538, 'epoch': 0.53}              
{'loss': 1.0558, 'grad_norm': 0.6947885155677795, 'learning_rate': 4.72e-05, 'num_tokens': 27899.0, 'mean_token_accuracy': 0.7316561937332153, 'epoch': 0.54}            
{'loss': 0.8572, 'grad_norm': 0.8561428785324097, 'learning_rate': 4.64e-05, 'num_tokens': 28319.0, 'mean_token_accuracy': 0.786057710647583, 'epoch': 0.54}             
{'loss': 0.853, 'grad_norm': 0.9124574661254883, 'learning_rate': 4.5600000000000004e-05, 'num_tokens': 28650.0, 'mean_token_accuracy': 0.7645260095596313, 'epoch': 0.55}
{'loss': 0.554, 'grad_norm': 0.9127132296562195, 'learning_rate': 4.4800000000000005e-05, 'num_tokens': 28902.0, 'mean_token_accuracy': 0.8629032373428345, 'epoch': 0.56}
{'loss': 0.8187, 'grad_norm': 0.8267902731895447, 'learning_rate': 4.4000000000000006e-05, 'num_tokens': 29323.0, 'mean_token_accuracy': 0.8129496574401855, 'epoch': 0.57}
{'loss': 0.8708, 'grad_norm': 0.8505646586418152, 'learning_rate': 4.32e-05, 'num_tokens': 29752.0, 'mean_token_accuracy': 0.7576470375061035, 'epoch': 0.58}            
{'loss': 1.1661, 'grad_norm': 0.7539921402931213, 'learning_rate': 4.24e-05, 'num_tokens': 30361.0, 'mean_token_accuracy': 0.6859503984451294, 'epoch': 0.58}            
{'loss': 1.0401, 'grad_norm': 0.7217645645141602, 'learning_rate': 4.16e-05, 'num_tokens': 30784.0, 'mean_token_accuracy': 0.7255370020866394, 'epoch': 0.59}            
{'loss': 1.1636, 'grad_norm': 0.7215142250061035, 'learning_rate': 4.08e-05, 'num_tokens': 31333.0, 'mean_token_accuracy': 0.6807339191436768, 'epoch': 0.6}             
{'loss': 0.9449, 'grad_norm': 0.6475540995597839, 'learning_rate': 4e-05, 'num_tokens': 31910.0, 'mean_token_accuracy': 0.7609075307846069, 'epoch': 0.61}               
{'loss': 0.944, 'grad_norm': 0.826789379119873, 'learning_rate': 3.9200000000000004e-05, 'num_tokens': 32315.0, 'mean_token_accuracy': 0.7581047415733337, 'epoch': 0.62}
{'loss': 0.7249, 'grad_norm': 0.7669816613197327, 'learning_rate': 3.8400000000000005e-05, 'num_tokens': 32689.0, 'mean_token_accuracy': 0.7972972989082336, 'epoch': 0.62}
{'loss': 1.2569, 'grad_norm': 0.8535155653953552, 'learning_rate': 3.76e-05, 'num_tokens': 33111.0, 'mean_token_accuracy': 0.7033492922782898, 'epoch': 0.63}            
{'loss': 0.6476, 'grad_norm': 1.0520843267440796, 'learning_rate': 3.68e-05, 'num_tokens': 33417.0, 'mean_token_accuracy': 0.8576158881187439, 'epoch': 0.64}            
{'loss': 1.0499, 'grad_norm': 0.7596135139465332, 'learning_rate': 3.6e-05, 'num_tokens': 33900.0, 'mean_token_accuracy': 0.7139874696731567, 'epoch': 0.65}             
{'loss': 0.8725, 'grad_norm': 0.8008524775505066, 'learning_rate': 3.52e-05, 'num_tokens': 34284.0, 'mean_token_accuracy': 0.7868421077728271, 'epoch': 0.66}            
{'loss': 1.0625, 'grad_norm': 0.5638991594314575, 'learning_rate': 3.4399999999999996e-05, 'num_tokens': 34924.0, 'mean_token_accuracy': 0.7154088020324707, 'epoch': 0.66}
{'loss': 0.8652, 'grad_norm': 0.806075394153595, 'learning_rate': 3.3600000000000004e-05, 'num_tokens': 35335.0, 'mean_token_accuracy': 0.7690417766571045, 'epoch': 0.67}
{'loss': 1.044, 'grad_norm': 0.7770567536354065, 'learning_rate': 3.2800000000000004e-05, 'num_tokens': 35757.0, 'mean_token_accuracy': 0.7368420958518982, 'epoch': 0.68}
{'loss': 0.694, 'grad_norm': 0.8995140194892883, 'learning_rate': 3.2000000000000005e-05, 'num_tokens': 36051.0, 'mean_token_accuracy': 0.8068965673446655, 'epoch': 0.69}
{'loss': 0.8615, 'grad_norm': 0.8023759722709656, 'learning_rate': 3.12e-05, 'num_tokens': 36408.0, 'mean_token_accuracy': 0.779036819934845, 'epoch': 0.7}              
{'loss': 0.5489, 'grad_norm': 0.8622695803642273, 'learning_rate': 3.04e-05, 'num_tokens': 36726.0, 'mean_token_accuracy': 0.8471337556838989, 'epoch': 0.7}             
{'loss': 0.8851, 'grad_norm': 1.0725154876708984, 'learning_rate': 2.96e-05, 'num_tokens': 37016.0, 'mean_token_accuracy': 0.8006992936134338, 'epoch': 0.71}            
{'loss': 0.9321, 'grad_norm': 1.1203888654708862, 'learning_rate': 2.88e-05, 'num_tokens': 37363.0, 'mean_token_accuracy': 0.7609329223632812, 'epoch': 0.72}            
{'loss': 1.2565, 'grad_norm': 0.7664346098899841, 'learning_rate': 2.8000000000000003e-05, 'num_tokens': 37881.0, 'mean_token_accuracy': 0.6712062358856201, 'epoch': 0.73}
{'loss': 1.1377, 'grad_norm': 0.8734380006790161, 'learning_rate': 2.7200000000000004e-05, 'num_tokens': 38231.0, 'mean_token_accuracy': 0.7803468108177185, 'epoch': 0.74}
{'loss': 1.1743, 'grad_norm': 0.7193783521652222, 'learning_rate': 2.64e-05, 'num_tokens': 38779.0, 'mean_token_accuracy': 0.6966911554336548, 'epoch': 0.74}            
{'loss': 0.8425, 'grad_norm': 0.978121280670166, 'learning_rate': 2.5600000000000002e-05, 'num_tokens': 39109.0, 'mean_token_accuracy': 0.8220859169960022, 'epoch': 0.75}
{'loss': 0.6135, 'grad_norm': 0.9066908359527588, 'learning_rate': 2.48e-05, 'num_tokens': 39449.0, 'mean_token_accuracy': 0.8303571343421936, 'epoch': 0.76}            
{'loss': 0.6754, 'grad_norm': 0.7623862028121948, 'learning_rate': 2.4e-05, 'num_tokens': 39833.0, 'mean_token_accuracy': 0.8184210658073425, 'epoch': 0.77}             
{'loss': 1.0122, 'grad_norm': 0.7582613229751587, 'learning_rate': 2.32e-05, 'num_tokens': 40218.0, 'mean_token_accuracy': 0.7427821755409241, 'epoch': 0.78}            
{'loss': 0.7705, 'grad_norm': 0.8276058435440063, 'learning_rate': 2.2400000000000002e-05, 'num_tokens': 40577.0, 'mean_token_accuracy': 0.8028169274330139, 'epoch': 0.78}
{'loss': 0.6144, 'grad_norm': 0.9646459817886353, 'learning_rate': 2.16e-05, 'num_tokens': 40915.0, 'mean_token_accuracy': 0.817365288734436, 'epoch': 0.79}             
{'loss': 0.8537, 'grad_norm': 1.0659465789794922, 'learning_rate': 2.08e-05, 'num_tokens': 41262.0, 'mean_token_accuracy': 0.7871720194816589, 'epoch': 0.8}             
{'loss': 1.5226, 'grad_norm': 0.8582617044448853, 'learning_rate': 2e-05, 'num_tokens': 41755.0, 'mean_token_accuracy': 0.6421267986297607, 'epoch': 0.81}               
{'loss': 1.2095, 'grad_norm': 0.7863159775733948, 'learning_rate': 1.9200000000000003e-05, 'num_tokens': 42158.0, 'mean_token_accuracy': 0.7117794752120972, 'epoch': 0.82}
{'loss': 0.897, 'grad_norm': 0.8209980130195618, 'learning_rate': 1.84e-05, 'num_tokens': 42505.0, 'mean_token_accuracy': 0.7609329223632812, 'epoch': 0.82}             
{'loss': 0.9518, 'grad_norm': 0.8437697887420654, 'learning_rate': 1.76e-05, 'num_tokens': 42851.0, 'mean_token_accuracy': 0.7660818696022034, 'epoch': 0.83}            
{'loss': 1.3841, 'grad_norm': 0.8331084251403809, 'learning_rate': 1.6800000000000002e-05, 'num_tokens': 43284.0, 'mean_token_accuracy': 0.687645673751831, 'epoch': 0.84}
{'loss': 1.3384, 'grad_norm': 0.5743568539619446, 'learning_rate': 1.6000000000000003e-05, 'num_tokens': 44154.0, 'mean_token_accuracy': 0.6454965472221375, 'epoch': 0.85}
{'loss': 0.7805, 'grad_norm': 0.7143769860267639, 'learning_rate': 1.52e-05, 'num_tokens': 44562.0, 'mean_token_accuracy': 0.7945544719696045, 'epoch': 0.86}            
{'loss': 1.0935, 'grad_norm': 0.7053938508033752, 'learning_rate': 1.44e-05, 'num_tokens': 45124.0, 'mean_token_accuracy': 0.7132616639137268, 'epoch': 0.86}            
{'loss': 0.8099, 'grad_norm': 0.7679741382598877, 'learning_rate': 1.3600000000000002e-05, 'num_tokens': 45455.0, 'mean_token_accuracy': 0.7737002968788147, 'epoch': 0.87}
{'loss': 0.6831, 'grad_norm': 1.0992088317871094, 'learning_rate': 1.2800000000000001e-05, 'num_tokens': 45729.0, 'mean_token_accuracy': 0.8259259462356567, 'epoch': 0.88}
{'loss': 1.1388, 'grad_norm': 0.8635069131851196, 'learning_rate': 1.2e-05, 'num_tokens': 46066.0, 'mean_token_accuracy': 0.7567567825317383, 'epoch': 0.89}             
{'loss': 0.972, 'grad_norm': 0.7276822328567505, 'learning_rate': 1.1200000000000001e-05, 'num_tokens': 46518.0, 'mean_token_accuracy': 0.71875, 'epoch': 0.9}           
{'loss': 1.3594, 'grad_norm': 0.9957514405250549, 'learning_rate': 1.04e-05, 'num_tokens': 46894.0, 'mean_token_accuracy': 0.6962365508079529, 'epoch': 0.9}             
{'loss': 0.8357, 'grad_norm': 0.8126235008239746, 'learning_rate': 9.600000000000001e-06, 'num_tokens': 47297.0, 'mean_token_accuracy': 0.7744361162185669, 'epoch': 0.91}
{'loss': 1.2183, 'grad_norm': 0.737899899482727, 'learning_rate': 8.8e-06, 'num_tokens': 47797.0, 'mean_token_accuracy': 0.6915322542190552, 'epoch': 0.92}              
{'loss': 0.7626, 'grad_norm': 0.899217963218689, 'learning_rate': 8.000000000000001e-06, 'num_tokens': 48132.0, 'mean_token_accuracy': 0.7885196208953857, 'epoch': 0.93}
{'loss': 0.9571, 'grad_norm': 0.6932201385498047, 'learning_rate': 7.2e-06, 'num_tokens': 48579.0, 'mean_token_accuracy': 0.7381489872932434, 'epoch': 0.94}             
{'loss': 0.9983, 'grad_norm': 0.6008664965629578, 'learning_rate': 6.4000000000000006e-06, 'num_tokens': 49320.0, 'mean_token_accuracy': 0.7272727489471436, 'epoch': 0.94}
{'loss': 0.8772, 'grad_norm': 0.7244109511375427, 'learning_rate': 5.600000000000001e-06, 'num_tokens': 49703.0, 'mean_token_accuracy': 0.7730870842933655, 'epoch': 0.95}
{'loss': 1.0119, 'grad_norm': 0.6550142168998718, 'learning_rate': 4.800000000000001e-06, 'num_tokens': 50175.0, 'mean_token_accuracy': 0.7393162250518799, 'epoch': 0.96}
{'loss': 0.9248, 'grad_norm': 0.9204407930374146, 'learning_rate': 4.000000000000001e-06, 'num_tokens': 50556.0, 'mean_token_accuracy': 0.7692307829856873, 'epoch': 0.97}
{'loss': 1.1977, 'grad_norm': 0.7627249360084534, 'learning_rate': 3.2000000000000003e-06, 'num_tokens': 51055.0, 'mean_token_accuracy': 0.7010101079940796, 'epoch': 0.98}
{'loss': 1.0087, 'grad_norm': 0.789191484451294, 'learning_rate': 2.4000000000000003e-06, 'num_tokens': 51487.0, 'mean_token_accuracy': 0.7383177280426025, 'epoch': 0.98}
{'loss': 1.2554, 'grad_norm': 0.699669361114502, 'learning_rate': 1.6000000000000001e-06, 'num_tokens': 51980.0, 'mean_token_accuracy': 0.6932515501976013, 'epoch': 0.99}
{'loss': 0.9506, 'grad_norm': 0.7264242768287659, 'learning_rate': 8.000000000000001e-07, 'num_tokens': 52435.0, 'mean_token_accuracy': 0.7516629695892334, 'epoch': 1.0}
{'train_runtime': 78.9798, 'train_samples_per_second': 6.331, 'train_steps_per_second': 1.583, 'train_loss': 1.009314311504364, 'epoch': 1.0}                            
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 125/125 [01:18<00:00,  1.58it/s]

============================================================
TRAINING COMPLETED
============================================================
Training runtime: 78.98 seconds
Samples per second: 6.33
Steps per second: 1.58
Train loss: 1.0093
============================================================

Saving the model
root@7d62de5c7fc1:/workspace/assets# 
