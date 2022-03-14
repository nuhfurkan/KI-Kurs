from xxlimited import new

import pandas
from model import Model
import pandas

features = [    
    "chocolate_color",
    "bake_time",
    "sugar",
    "flour",
    "margerine",
    "chocolate",
    "vanilla_flavour",
    "almonds",
    "density",
    "brown_sugar",
    "weight",
    "cinnamon"
]      

df = pandas.read_csv("cookies.csv")


print("start")
newModel = Model()

newModel.fit(df[features], df["taste"])

test = [
    [0,14.182,110.667,118.072,101.534,106.478,0.451,14.401,0.619,41.085,6.938,3.623],
    [0,13.455,110.0,125.301,110.429,105.814,0.955,21.083,0.668,38.76,7.191,1.884],
    [1,9.455,129.333,109.036,102.301,107.475,0.122,6.797,0.621,91.473,7.865,7.391],
    [0,13.273,108.0,114.458,109.049,105.814,0.469,10.829,0.623,31.008,5.927,3.188]
]
print(newModel.predict(test))