import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
import io
import base64

# PRIMERO: Configurar la página
st.set_page_config(
    page_title="Predicción de PCI", 
    layout="wide",
    page_icon="🛣️",
    initial_sidebar_state="expanded"
)

# Cargar modelo y scaler
@st.cache_resource
def load_model_and_scaler():
    with open("catboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("standard_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

@st.cache_resource
def load_shap_explainer(_model):
    """Cargar el explainer de SHAP (cached para mejor performance)"""
    explainer = shap.TreeExplainer(_model)
    return explainer

model, scaler = load_model_and_scaler()
explainer = load_shap_explainer(model)

# Lista de variables esperadas
feature_names = [
    "ini_pci", "edad_tupla", "CDD_WAVG", "CWD_WAVG", "Rx1day_Mar", "Rx1day_Apr", 
    "Rx1day_Nov", "Rx5day_Feb", "Rx5day_May", "Rx5day_Jun", "Rx5day_Jul", "Rx5day_Aug", 
    "Rx5day_Sep", "Rx5day_Dec", "ID_WAVG", "Tx35_WAVG", "TXn_Apr", "DTR_Dec", "SN_VALUE", 
    "ANL_KESAL_LTPP_LN_YR_WAVG", "AADT_ALL_VEHIC_WAVG", "AADT_TRUCK_COMBO_WAVG"
]

# Función para crear gráfico SHAP waterfall
def create_shap_waterfall(shap_values, expected_value, feature_values_scaled, feature_values_original, feature_names):
    """Crear un gráfico waterfall de SHAP values usando Plotly"""
    
    # Preparar datos para el waterfall
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Ordenar por valor absoluto de SHAP
    indices = np.argsort(np.abs(shap_vals))[::-1]
    
    # Tomar top 10 features más importantes
    top_indices = indices[:10]
    top_shap_vals = shap_vals[top_indices]
    top_features = [feature_names[i] for i in top_indices]
    top_feature_vals_original = [feature_values_original[i] for i in top_indices]
    
    # Crear labels con valores originales (desnormalizados)
    labels = [f"{feat}<br>({val:.2f})" for feat, val in zip(top_features, top_feature_vals_original)]
    
    # Invertir el orden para mostrar de mayor a menor impacto (de arriba hacia abajo)
    labels = labels[::-1]
    top_shap_vals = top_shap_vals[::-1]
    
    # Preparar datos para waterfall
    colors = ['blue' if x < 0 else 'red' for x in top_shap_vals]
    
    fig = go.Figure()
    
    # Agregar barras horizontales
    fig.add_trace(go.Bar(
        x=top_shap_vals,
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{x:+.2f}" for x in top_shap_vals],
        textposition='outside',
        name='SHAP Values'
    ))
    
    fig.update_layout(
        title=f"🔍 Explicación de la Predicción (SHAP)<br>Top 10 variables que más impactan | Valor base: {expected_value:.2f}",
        xaxis_title="Contribución al PCI",
        yaxis_title="Variables (valor actual)",
        height=500,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

# Función para crear gráfico de gauge
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "red"},
                {'range': [25, 50], 'color': "orange"},
                {'range': [50, 75], 'color': "yellow"},
                {'range': [75, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

# Función para interpretar el PCI
def interpret_pci(pci_value):
    if pci_value >= 85:
        return "🟢 Excelente", "El pavimento está en condiciones excelentes"
    elif pci_value >= 70:
        return "🟡 Bueno", "El pavimento está en buenas condiciones con mantenimiento menor"
    elif pci_value >= 55:
        return "🟠 Regular", "El pavimento requiere mantenimiento preventivo"
    elif pci_value >= 40:
        return "🔴 Malo", "El pavimento requiere rehabilitación"
    else:
        return "⚫ Muy Malo", "El pavimento requiere reconstrucción"

def create_deterioration_curve(user_input, model, scaler, feature_names):
    """Crear curva de deterioro del pavimento"""
    
    # Calcular años desde edad_tupla
    edad_dias = user_input['edad_tupla']
    años = max(1, int(edad_dias / 365))  # Mínimo 1 año
    
    # Datos para la curva
    años_lista = []
    pci_lista = []
    
    # Punto inicial (año 0): ini_pci
    años_lista.append(0)
    pci_lista.append(user_input['ini_pci'])
    
    # Crear predicciones para cada año intermedio
    for año in range(1, años + 1):  # Incluir el año actual
        # Copiar datos originales
        datos_año = user_input.copy()
        # Cambiar edad_tupla
        datos_año['edad_tupla'] = año * 365
        
        # Predecir
        input_df = pd.DataFrame([datos_año])
        input_scaled = scaler.transform(input_df)  # Sin [feature_names]
        prediccion = model.predict(input_scaled)[0]
        
        años_lista.append(año)
        pci_lista.append(prediccion)
    
    # Crear gráfico
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=años_lista,
        y=pci_lista,
        mode='lines+markers',
        name='Curva de Deterioro',
        line=dict(color='red', width=3),
        marker=dict(size=8, color='darkred')
    ))
    
    fig.update_layout(
        title="📉 Curva de Deterioro del Pavimento",
        xaxis_title="Años",
        yaxis_title="PCI",
        height=400,
        showlegend=False
    )
    
    return fig

# Header con estilo
st.markdown("""
<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; text-align: center; margin: 0;">
        🛣️ Predictor Inteligente de PCI
    </h1>
    <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
        Índice de Condición del Pavimento usando Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar para seleccionar modo
st.sidebar.header("⚙️ Configuración")
input_mode = st.sidebar.radio(
    "Selecciona el método de entrada:",
    ["📁 Cargar archivo (CSV/Excel)", "✏️ Entrada manual"]
)

# Información en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre el modelo")
st.sidebar.info("""
Este modelo utiliza **CatBoost** para predecir el PCI basándose en:
- Condiciones climáticas
- Características del pavimento
- Tráfico 
""")

# Contenido principal
if input_mode == "📁 Cargar archivo (CSV/Excel)":
    st.markdown("### 📂 Carga tu archivo de datos")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Arrastra y suelta tu archivo aquí",
            type=['csv', 'xlsx', 'xls'],
            help="El archivo debe contener las columnas necesarias para la predicción"
        )
    
    with col2:
        st.markdown("#### 📋 Columnas requeridas:")
        with st.expander("Ver lista completa"):
            for i, feature in enumerate(feature_names, 1):
                st.write(f"{i}. {feature}")
    
    if uploaded_file is not None:
        try:
            # Leer archivo según extensión
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Archivo cargado exitosamente: {uploaded_file.name}")
            
            # Mostrar vista previa
            st.markdown("### 👀 Vista previa de los datos")
            st.dataframe(df.head(), use_container_width=True)
            
            # Verificar columnas
            missing_cols = [col for col in feature_names if col not in df.columns]
            if missing_cols:
                st.error(f"❌ Faltan las siguientes columnas: {missing_cols}")
            else:
                st.success("✅ Todas las columnas requeridas están presentes")
                
                # Botón para hacer predicciones
                if st.button("🚀 Realizar predicciones", type="primary"):
                    # Preparar datos
                    input_data = df[feature_names]
                    
                    # Normalizar
                    input_scaled = scaler.transform(input_data)
                    
                    # Predecir
                    predictions = model.predict(input_scaled)
                    
                    # Agregar predicciones al DataFrame
                    df['PCI_Predicho'] = predictions
                    
                    # Mostrar resultados
                    st.markdown("### 📈 Resultados de las predicciones")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicciones realizadas", len(predictions))
                    with col2:
                        st.metric("PCI Predicho Promedio", f"{predictions.mean():.2f}")
                    with col3:
                        st.metric("Desviación Estándar de las Predicciones", f"{predictions.std():.2f}")
                    
                    # Gráfico de distribución - Solo si hay múltiples predicciones
                    if len(predictions) > 1:
                        fig = px.histogram(
                            x=predictions, 
                            nbins=20,
                            title="Distribución de Predicciones PCI",
                            labels={'x': 'PCI Predicho', 'y': 'Frecuencia'}
                        )
                        fig.update_layout(showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("📊 El histograma se muestra cuando hay múltiples predicciones")
                    
                    # SHAP y CURVAS DE DETERIORO - Solo para 5 o menos predicciones
                    if len(predictions) <= 5:
                        st.markdown("### 🔍 Explicación SHAP de las predicciones")
                        
                        # Calcular SHAP values
                        shap_values = explainer.shap_values(input_scaled)
                        expected_value = explainer.expected_value
                        
                        # Mostrar explicación para cada predicción
                        for i in range(len(predictions)):
                            with st.expander(f"📊 Explicación predicción #{i+1} (PCI: {predictions[i]:.2f})"):
                                shap_fig = create_shap_waterfall(
                                    shap_values[i], 
                                    expected_value, 
                                    input_scaled[i], 
                                    input_data.iloc[i].values,  # Valores originales
                                    feature_names
                                )
                                st.plotly_chart(shap_fig, use_container_width=True)
                        
                        # CURVAS DE DETERIORO para cada fila
                        st.markdown("### 📉 Curvas de Deterioro")
                        
                        for i in range(len(predictions)):
                            with st.expander(f"📈 Curva de deterioro #{i+1} (PCI: {predictions[i]:.2f})"):
                                # Convertir fila a diccionario
                                user_input_from_file = input_data.iloc[i].to_dict()
                                
                                deterioration_fig = create_deterioration_curve(
                                    user_input_from_file, model, scaler, feature_names
                                )
                                st.plotly_chart(deterioration_fig, use_container_width=True)
                    
                    else:
                        st.warning("⚠️ **Análisis detallado limitado**: El análisis SHAP y las curvas de deterioro solo se muestran para archivos con 5 o menos predicciones.")
                    
                    # Tabla de resultados
                    st.markdown("### 📋 Tabla de resultados")
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Descargar resultados
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar resultados (CSV)",
                        data=csv,
                        file_name="predicciones_pci.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error al procesar el archivo: {str(e)}")

else:  # Entrada manual
    st.markdown("### ✏️ Entrada manual de datos")
    
    # Inicializar valores por defecto si no existen (ANTES de crear widgets)
    for feature in feature_names:
        if f"input_{feature}" not in st.session_state:
            st.session_state[f"input_{feature}"] = 0.0
    
    # Organizar campos en categorías
    categories = {
        "🏗️ Datos del pavimento": ["ini_pci", "edad_tupla", "SN_VALUE"],
        "🌡️ Datos climáticos": [col for col in feature_names if any(x in col for x in ["CDD", "CWD", "Rx", "ID", "Tx", "DTR"])],
        "🚛 Datos de tráfico": ["ANL_KESAL_LTPP_LN_YR_WAVG", "AADT_ALL_VEHIC_WAVG", "AADT_TRUCK_COMBO_WAVG"]
    }
    
    # Verificar si hay una predicción activa
    prediction_active = st.session_state.get('prediction_made', False)
    
    # Funciones de callback para limpiar (solo funcionan si no hay predicción)
    def clear_field(field_name):
        if not st.session_state.get('prediction_made', False):
            st.session_state[f"input_{field_name}"] = 0.0
    
    def clear_category_fields(fields):
        if not st.session_state.get('prediction_made', False):
            for field in fields:
                st.session_state[f"input_{field}"] = 0.0
    
    def clear_prediction():
        # Solo eliminar la predicción, mantener valores
        if 'prediction_made' in st.session_state:
            del st.session_state['prediction_made']
        if 'user_input_data' in st.session_state:
            del st.session_state['user_input_data']
        if 'prediction_result' in st.session_state:
            del st.session_state['prediction_result']
    
    # Crear tabs para cada categoría
    tabs = st.tabs(list(categories.keys()))
    
    for tab, (category, fields) in zip(tabs, categories.items()):
        with tab:
            # Botón para limpiar toda la categoría (deshabilitado si hay predicción)
            col_clear = st.columns([4, 1])
            with col_clear[1]:
                # Mapeo correcto de categorías
                category_names = {
                    "🏗️ Datos del pavimento": "pavimento",
                    "🌡️ Datos climáticos": "climáticos", 
                    "🚛 Datos de tráfico": "tráfico"
                }
                category_key = category_names[category]
                st.button(
                    f"🗑️ Limpiar {category_key}", 
                    key=f"clear_cat_{category_key}", 
                    help=f"Limpiar todos los campos de {category_key}" if not prediction_active else "Elimina la predicción para habilitar",
                    disabled=prediction_active,
                    on_click=clear_category_fields,
                    args=(fields,)
                )
            
            # Campos de entrada
            cols = st.columns(2)
            for i, feature in enumerate(fields):
                with cols[i % 2]:
                    # Crear columnas para el campo y el botón X
                    field_col, btn_col = st.columns([4, 1])
                    
                    with field_col:
                        st.number_input(
                            label=feature.replace('_', ' ').title(),
                            key=f"input_{feature}",
                            help=f"Ingresa el valor para {feature}",
                            disabled=prediction_active
                        )
                    
                    with btn_col:
                        st.markdown("<br>", unsafe_allow_html=True)  # Espaciado
                        st.button(
                            "🧹", 
                            key=f"clear_{feature}", 
                            help=f"Limpiar {feature}" if not prediction_active else "Elimina la predicción para habilitar",
                            disabled=prediction_active,
                            on_click=clear_field,
                            args=(feature,)
                        )
    
    # Botones de predicción y eliminar predicción
    st.markdown("---")
    
    if prediction_active:
        # Si hay predicción, mostrar botón para eliminarla
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.button(
                "🗑️ Eliminar predicción", 
                type="secondary", 
                use_container_width=True,
                help="Eliminar la predicción actual para modificar valores",
                on_click=clear_prediction
            )
    else:
        # Si no hay predicción, mostrar botón para predecir
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🎯 Predecir PCI", type="primary", use_container_width=True)
        
        if predict_button:
            # Recopilar valores de los number_input widgets
            user_input = {}
            for feature in feature_names:
                user_input[feature] = st.session_state.get(f"input_{feature}", 0.0)
            
            # Guardar datos en session_state
            st.session_state['prediction_made'] = True
            st.session_state['user_input_data'] = user_input
            
            # Convertir a DataFrame
            input_df = pd.DataFrame([user_input])
            
            # Normalizar
            input_scaled = scaler.transform(input_df)
            
            # Predecir
            prediction = model.predict(input_scaled)[0]
            
            # Guardar resultado
            st.session_state['prediction_result'] = prediction
            
            st.rerun()
    
    # Mostrar resultados si hay una predicción activa
    if prediction_active and 'prediction_result' in st.session_state:
        prediction = st.session_state['prediction_result']
        user_input = st.session_state['user_input_data']
        
        # Mostrar resultado con estilo
        st.markdown("---")
        st.markdown("### 🎉 Resultado de la predicción")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Gauge chart
            gauge_fig = create_gauge_chart(prediction, "PCI Predicho")
            st.plotly_chart(gauge_fig, use_container_width=True)
        
        with col2:
            # Interpretación
            status, description = interpret_pci(prediction)
            st.markdown(f"### {status}")
            st.markdown(f"**Valor predicho:** {prediction:.2f}")
            st.markdown(f"**Interpretación:** {description}")
            
            # Recomendaciones
            if prediction >= 70:
                st.success("✅ Mantener el programa de mantenimiento actual")
            elif prediction >= 55:
                st.warning("⚠️ Considerar mantenimiento preventivo")
            else:
                st.error("🚨 Requiere intervención inmediata")
        
        # EXPLICACIÓN SHAP
        st.markdown("### 🔍 ¿Por qué esta predicción?")
        st.markdown("**Análisis SHAP:** Descubre qué factores más influyeron en el resultado")
        
        # Recalcular para mostrar
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        
        # Calcular SHAP values para la predicción individual
        shap_values = explainer.shap_values(input_scaled)
        expected_value = explainer.expected_value
        
        # Crear gráfico SHAP waterfall
        shap_fig = create_shap_waterfall(
            shap_values[0], 
            expected_value, 
            input_scaled[0], 
            list(user_input.values()),  # Valores originales desnormalizados
            feature_names
        )
        st.plotly_chart(shap_fig, use_container_width=True)
        
        # Explicación de SHAP
        st.info("""
        💡 **Cómo leer este gráfico:**
        - **Barras rojas**: Factores que AUMENTAN el PCI
        - **Barras azules**: Factores que DISMINUYEN el PCI  
        - **Valor base**: Predicción promedio del modelo
        - **Números**: Contribución de cada variable al resultado final
        """)
        
        # CURVA DE DETERIORO
        st.markdown("### 📉 Curva de Deterioro")
        deterioration_fig = create_deterioration_curve(user_input, model, scaler, feature_names)
        st.plotly_chart(deterioration_fig, use_container_width=True)

        # Detalles técnicos
        with st.expander("🔍 Ver detalles técnicos"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Datos de entrada normalizados:**")
                normalized_df = pd.DataFrame(input_scaled, columns=feature_names)
                st.dataframe(normalized_df.T, use_container_width=True)
            with col2:
                st.markdown("**Datos originales:**")
                st.dataframe(input_df.T, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 Desarrollado con Streamlit y CatBoost | 📊 Modelo de Machine Learning para predicción de PCI</p>
</div>
""", unsafe_allow_html=True)