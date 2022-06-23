## EMNLP submission  

## Build and run the project  
### Build
`docker build -t gpt_bert_gan .`  

### Dataset selction:  
The second agument of run_gpt_dpgan.py is gpu id and the first is the dataset selection(same for other commands)  
1: image coco (base dataset for the following commands)  
2: EMNLP news  
3: EMNLP news + coco dataset  
4: EMNLP news reduced (a reduced version of EMNLP news)  

### Train gpt_dpgan (dpgan with GPT-2 generator)
```
docker run --gpus all --rm -v "$(pwd)"/src:/app gpt_bert_gan bash -c "cd /app/run && python -m run_gpt_dpgan.py 1 2"
```  

### Train gpt_bert_dpgan (gpt-2 with BERT trained for sentiment analysis)  
```
docker run --gpus all --rm -v "$(pwd)"/src:/app gpt_bert_gan bash -c "cd /app/run && python -m run_gpt_bert_dpgan.py 1 2"
```

### Train gpt_bert_fake_gan (gpt-2 with BERT trained for fake detection)
```
docker run --gpus all --rm -v "$(pwd)"/src:/app gpt_bert_gan bash -c "cd /app/run && python -m run_gpt_bert_fake_gan.py 1 2"
```

## Plots and visuals  
All code to create the plots can be found at [visual/training_plots.py](./src/visual/training_plots.py) and the plots produced will be put in folder [visual/saved_plots](./src/visual/saved_plots)     

### Visuals for gpt-dpgan  
If you use "run_gpt_dpgan.py 1 2" you can choose to display the negative reward using the parameter --show_rewards   
```
docker run --gpus all --rm -v "$(pwd)"/src:/app gpt_bert_gan bash -c "cd /app/run && python -m run_gpt_dpgan.py 1 2 --show_rewards=True"
```
If you use use "run_gpt_dpgan.py 1 2" you can choose to display the most probable token at position using the parameter --show_probs  
```
docker run --gpus all --rm -v "$(pwd)"/src:/app gpt_bert_gan bash -c "cd /app/run && python -m run_gpt_dpgan.py 1 2 --show_probs=True"
```

### Sentiment plots for gpt_bert_dpgan  
An histogram is produced automatically at the end of training at [visual/saved_plots](./src/visual/saved_plots)  
You also have at each epoch the number of positive/neutral/negative samples that can be used to produce a plot using plot_negativity_evolution function in visual/training_plots.py  
You can use the parameter --run_validation to compare base gpt-2 and trained gpt-2 on another validation dataset, which will produce an histogram in [visual/saved_plots](./src/visual/saved_plots). For the trained gpt-2, it is loaded from [pretrained](./src/pretrained) folder. You can train another model(saved models are stored automatically in [save](./src/save)) or download the model here https://huggingface.co/b5ac05bb256aaeeb4283ce8379ad71d1/gpt2-nice/tree/main and use the exact same name, and put it in pretrained/emnlp_news.  
```
docker run --gpus all --rm -v "$(pwd)"/src:/app gpt_bert_gan bash -c "cd /app/run && python -m run_gpt_bert_dpgan.py 2 2 --run_validation=True"
```

### Fake/True plots for gpt_bert_fake_gan  
You can get all training info for the discriminator (loss, accuracy and other metrics) using wandb and the parameter of Trainer to use wandb in [models/BERT_fake.py](./src/models/BERT_fake.py).      


## Training logs  
Training is logged at the [save](./src/save) folder + [log](./src/log) folder

## Repository structure  
[instructor](./src/instructor/real_data) is the folder where resides the code for training  
[models](./src/models) is the folder where resides the code for generators and discriminators  
[utils](./src/utils) is the folder where resides code to load the data from the datasets and process the data  
[visual](./src/visual) is the folder where resides the code for producing plots  
[config.py](./src/config.py) and [run](./src/run) folder are for configuration of the different trainings  



## Original repo
https://github.com/williamSYSU/TextGAN-PyTorch




