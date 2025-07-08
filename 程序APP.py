import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import xgboost
from xgboost import XGBClassifier


# 加载保存的随机森林模型
model = joblib.load('XGBoost.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "WBC": {"type": "numerical", "min": 0.000, "max": 570.000, "default": 5.000},
    "CRP": {"type": "numerical", "min": 0.000, "max": 690.000, "default": 1.000},
    "IL6": {"type": "numerical", "min": 0.000, "max": 700.000, "default": 1.500},
    "PCT": {"type": "numerical", "min": 0.000, "max": 200.000, "default": 0.050},
    "Elderly": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Fever status": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Restricted antimicrobial use": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Urinary catheterization": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Special-class antimicrobial use": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Antimicrobial use": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Bacterial infection": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Combination antimicrobial therapy": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Central venous catheter (CVC)": {"type": "categorical", "options": [0, 1], "default": 0, "mapping": {0: "否", 1: "是"}},
    "Disease type": {
        "type": "categorical",
        "options": [1, 2, 3],
        "default": 1,
        "mapping": {
            0: "Others",
            1: "Rspiratory disease",
            2: "Tumour",
            3: "Gynecological disease",
            4: "Orthopedics and Trauma",
            6: "CVD",
            7: "Nerve system disease",
            8: "Digestive system disease",
            9: "Metabolic diseases",
            10: "Urinary system disease",
            11: "Infectious disease",
            12: "Otorhinolaryngological diseases",
            13: "Ophthalmic Diseases",
            14: "Dermatological diseases",
            15: "Hematological diseases",
            16: "Rehabilitation",
            17: "ICU",
            18: "Gereology",
            19: "General medicine",
            20: "Traditional Chinese Medicine and General Medicine" 
        }
    }
}

# Streamlit 界面
import streamlit as st



# 添加其他内容
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
