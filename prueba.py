import streamlit as st
from datetime import datetime
import logging
import pickle
import joblib
import os

if not os.path.exists('logs'):
    os.makedirs('logs') 

logging.basicConfig(filename='logs/app.logs', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger()

logger.info("Application started")

def load_model():
    
    model = joblib.load('linear_regression_model.pkl')
    return model


model=None
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:  
    st.error(f"Error loading model: {e}") 
    st.stop()
    raise e

st.title("Concrete Strength Predictor")

st.sidebar.header("Input Features")
st.sidebar.markdown("Adjust the number_inputs to set the input features for predicting concrete strength.")

cement=st.sidebar.number_input("Cement",  min_value=100.0, max_value=500.0, value=300.0)
blastfurnace=st.sidebar.number_input("Blast Furnace Slag",  min_value=0.0, max_value=300.0, value=100.0)
flyash=st.sidebar.number_input("Fly Ash",  min_value=0.0, max_value=200.0, value=50.0)
water=st.sidebar.number_input("Water",  min_value=100.0, max_value=300.0, value=200.0)
superplasticizer=st.sidebar.number_input("Superplasticizer",  min_value=0.0, max_value=50.0, value=10.0)
coarseaggregate=st.sidebar.number_input("Coarse Aggregate",  min_value=800.0, max_value=1200.0, value=1000.0)
fineaggregate=st.sidebar.number_input("Fine Aggregate",  min_value=500.0, max_value=900.0, value=700.0)
age=st.sidebar.number_input("Age",  min_value=1, max_value=365, value=28)
concrete_compressive_strength=st.sidebar.number_input("Concrete Compressive Strength",  min_value=0.0, max_value=100.0, value=30.0)

if st.button("Calcular fuerza del concreto"):
    if cement <= 0 or blastfurnace < 0 or flyash < 0 or water <= 0 or superplasticizer < 0 or coarseaggregate <= 0 or fineaggregate <= 0 or age <= 0:
        st.error("Please enter valid positive values for all input features.")
        logger.warning("Invalid input values provided.")
        st.stop()
    if cement==None or blastfurnace==None or flyash==None or water==None or superplasticizer==None or coarseaggregate==None or fineaggregate==None or age==None:
        st.error("Please enter all input features.")
        logger.warning("Missing input values.")
        st.stop()
    if model:
        start_time = datetime.now()
        input_data = [[cement, blastfurnace, flyash, water, superplasticizer, coarseaggregate, fineaggregate, age]]
        prediction = model.predict(input_data)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Prediction made:{prediction[0]:.2f} MPa in {duration:.4f} seconds")
        st.success(f"Predicted Concrete Strength: {prediction[0]:.2f} MPa")
    else:
        st.error(f"Error during prediction")
    
st.markdown("---")
st.sidebar.subheader("Estadisticas del Modelo")
def get_model_stats(model):
    try:
        with open('logs/app.logs', 'rb') as f:
            logs = f.readlines()
            
        today = datetime.now().date()
        latencys = [l for l in logs if "Prediction made" in l.decode('utf-8')]
        avg_latency = sum([float(l.decode('utf-8').split("in")[-1].strip().split(" ")[0]) for l in latencys]) / len(latencys) if latencys else 0.0
        
        
        return {
            "Total Predictions": len(latencys),
            "Average Strength": avg_latency,
        }
    
    except Exception as e:
        st.error(f"Error loading model statistics: {e}")
        return None 
    
stats = get_model_stats(model)
st.sidebar.write(f"Total Predictions: {stats['Total Predictions']}")
st.sidebar.write(f"Fuerza del concreto: {stats['Average Strength']:.2f} MPa")
#st.sidebar.write(f"Average Latency (seconds): {stats['Average Latency (seconds)']:.4f}") 
    