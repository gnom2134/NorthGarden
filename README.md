# NorthGarden

## Get preprocessing to Google Colab

Type following command in colab cell
```shell
!bash <(curl -s "https://raw.githubusercontent.com/gnom2134/NorthGarden/preprocessing/colab_settings.sh")
```
Then you will be able to import preprocessing functions and access the data at ```./SouthParkData/All-seasons.csv```.

For example:
```python
from inputs import df_cleanup, generator_input

df = pd.read_csv(Path("./SouthParkData/All-seasons.csv"))
df_cleanup(df)
gen_input = generator_input(df, ["Cartman"], n_context=2)

print(gen_input["Cartman"][0])
print(gen_input["Cartman"][1])
```