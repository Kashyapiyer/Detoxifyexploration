Simple exploration with Detoxify 

References:
* https://github.com/unitaryai/detoxify

Datasets: 

* https://huggingface.co/datasets/Johnesss/Jigsaw-Toxic-Comment-Classification/
* https://github.com/surge-ai/toxicity/blob/main/toxicity_en.csv

#filtered_df = testsample[testsample['is_toxic'].isin(['Toxic','Not Toxic'])].sample(n=20, random_state=1).reset_index(drop=True)
