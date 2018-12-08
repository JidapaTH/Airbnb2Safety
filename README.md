# Predicting Neighborhood Safety using Airbnb data
Winnie Gao, Fei Ren, Somsakul Somboon,  Jidapa Thanabhusest, Jing Wang

We hope to add another dimension to crime maps, the perception of the neighborhood safety levels from visitors and sub-letters, to make crime maps more comprehensive and representative. New residents and tourists can compare our crime maps with crime maps made with police data of the same area to make a more informed decision on where they want to stay. People living or traveling to cities where there’s no public crime data can be mostly benefited from our work.

We trained our models using open source crime and Airbnb data from large cities including Los Angeles and Austin. Our work consists four major parts which are data cleaning, crime score prediction model, feature generation by NLP and  data modeling. The major challenges were generating crime score and model features at neighborhood level which requires aggregating different information. Modeling robustness was also limited by the entropies of airbnb text review data. Therefore we trained most promising multiclass classifiers to improve prediction performance.  Our final output is the interactive choropleth map which allows users to locate areas by zipcode and view safety levels predicted by our best trained model.  

## Architecture
<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/archi.GIF"  width="50%" height="50%" ></p>

## Crime Score
We analyzed the crime report dataset using multiple approaches in getting the crime score. We first start out by intuitively ranking the crime, the improved version is referencing the sentencing years before finally settling with a wider spectrum ranging from 0-100 based on crime categories

## Data Modeling

<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/model.GIF"  width="50%" height="50%" ></p>

## NLP Features
<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/NLP.GIF"  width="50%" height="50%" ></p>


Some important words
<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/word.GIF"  width="50%" height="50%" ></p>



## Result
We have relatively small sample size but over hundred features so we used cross validation. We chosen these models based on our multi-class classification problem and account for the fact that we have mixed predictors.  We found that Adaboost and SVM classifiers consistently performs the best. Overall, our model increases prediction accuracy from baseline model by near 15%.

The significant features
<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/features.GIF"  width="50%" height="50%" ></p>

<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/Pred_SearchBar_Map.png"  width="50%" height="50%" ></p>
<p align="center"> <img src="https://github.com/JidapaTH/Airbnb2Safety/blob/master/Error_Map.png"  width="50%" height="50%" ></p>







