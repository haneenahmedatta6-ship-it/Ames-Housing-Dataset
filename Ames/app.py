import gradio as gr
import pandas as pd
import joblib

# Load the saved preprocessing + model pipeline
model = joblib.load("ames_house_model.pkl")

def predict_price(overall_qual, living_area, basement_area, garage_cars,
                  year_built, lot_area, bedrooms, neighborhood,
                  house_style, exterior_quality, kitchen_quality, garage_type):
    input_data = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": living_area,
        "TotalBsmtSF": basement_area,
        "GarageCars": garage_cars,
        "YearBuilt": year_built,
        "LotArea": lot_area,
        "BedroomAbvGr": bedrooms,
        "Neighborhood": neighborhood,
        "HouseStyle": house_style,
        "ExterQual": exterior_quality,
        "KitchenQual": kitchen_quality,
        "GarageType": garage_type
    }])
    prediction = model.predict(input_data)[0]
    return f"Estimated house price: ${prediction:,.0f}"

demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Slider(1, 10, value=6, step=1, label="Overall Quality (1-10)"),
        gr.Number(value=1500, label="Above-ground living area (sq ft)"),
        gr.Number(value=800, label="Basement area (sq ft)"),
        gr.Slider(0, 4, value=2, step=1, label="Garage capacity (cars)"),
        gr.Slider(1870, 2010, value=2000, step=1, label="Year built"),
        gr.Number(value=8000, label="Lot area (sq ft)"),
        gr.Slider(0, 8, value=3, step=1, label="Bedrooms above ground"),
        gr.Dropdown(["CollgCr","Veenker","Crawfor","NoRidge","Mitchel","Somerst","NWAmes","OldTown","BrkSide","Sawyer","NridgHt","NAmes","SawyerW","IDOTRR","MeadowV","Edwards","Timber","Gilbert","StoneBr","ClearCr","NPkVill","Blmngtn","BrDale","SWISU","Blueste"], value="CollgCr", label="Neighborhood"),
        gr.Dropdown(["1Story","2Story","1.5Fin","1.5Unf","SFoyer","SLvl","2.5Unf","2.5Fin"], value="2Story", label="House style"),
        gr.Dropdown(["Ex","Gd","TA","Fa","Po"], value="Gd", label="Exterior quality"),
        gr.Dropdown(["Ex","Gd","TA","Fa","Po"], value="Gd", label="Kitchen quality"),
        gr.Dropdown(["Attchd","Detchd","BuiltIn","Basment","CarPort","2Types","No Garage"], value="Attchd", label="Garage type")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Ames House Price Prediction",
    description="Enter house details to estimate its sale price using a trained machine-learning model."
)

if __name__ == "__main__":
    demo.launch()
