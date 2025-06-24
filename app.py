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
from catboost import CatBoostRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Definición de la clase CatBoostPCIModel (NECESARIA para pickle)
FEATURE_COLUMNS = [
    'ini_pci',
    'edad_tupla',
    'TOTAL_PRECIP_WAVG',
    'CDD_WAVG',
    'CWD_WAVG',
    'Rx1day_Mar',
    'Rx1day_Apr',
    'Rx1day_Oct',
    'Rx1day_Nov',
    'Rx5day_May',
    'Rx5day_Jun',
    'Rx5day_Jul',
    'Rx5day_Dec',
    'ID_WAVG',
    'Tx35_WAVG',
    'TXn_Apr',
    'DTR_Dec',
    'SN_VALUE',
    'ANL_KESAL_LTPP_LN_YR_WAVG',
    'AADT_ALL_VEHIC_WAVG',
    'AADT_TRUCK_COMBO_WAVG'
]

TARGET_COLUMN = 'fin_pci'

class CatBoostPCIModel:
    """
    Réplica del pipeline PyCaret con CatBoost - características fijas
    """

    def __init__(self):
        # 1. Imputador numérico (estrategia: media)
        self.numerical_imputer = SimpleImputer(strategy='mean')

        # 2. Normalizador MinMax
        self.scaler = MinMaxScaler()

        # 3. Modelo CatBoost con configuración de PyCaret
        self.model = CatBoostRegressor(
            # Configuración básica de PyCaret
            loss_function='RMSE',
            border_count=254,
            verbose=False,
            task_type='CPU',
            random_state=42,

            # Parámetros adicionales para mayor similitud
            iterations=1000,
            learning_rate=0.05598299950361252,
            depth=6,
            l2_leaf_reg=3,
            subsample=0.800000011920929,
            bootstrap_type='MVS',
            sampling_frequency='PerTree',
            max_leaves=64,
            min_data_in_leaf=1,
            grow_policy='SymmetricTree',
            leaf_estimation_method='Newton',
            leaf_estimation_iterations=1,
            leaf_estimation_backtracking='AnyImprovement',
            feature_border_type='GreedyLogSum',
            penalties_coefficient=1,
            model_size_reg=0.5,
            model_shrink_mode='Constant',
            model_shrink_rate=0,
            random_strength=1,
            random_score_type='NormalWithModelSizeDecrease',
            rsm=1,
            eval_metric='RMSE',
            boosting_type='Plain',
            score_function='Cosine',
            nan_mode='Min',
            boost_from_average=True,
            use_best_model=False,
            best_model_min_trees=1,
            sparse_features_conflict_fraction=0
        )

        # Variables de control
        self.is_fitted = False

    def prepare_data(self, dataset):
        """Preparar datos con las características fijas del pipeline"""
        # Verificar que todas las columnas estén presentes
        missing_features = [col for col in FEATURE_COLUMNS if col not in dataset.columns]
        if missing_features:
            raise ValueError(f"Columnas faltantes: {missing_features}")

        # Seleccionar variables especificadas
        X = dataset[FEATURE_COLUMNS].copy()
        y = dataset[TARGET_COLUMN].copy()

        print(f"Usando {len(FEATURE_COLUMNS)} características fijas")

        return X, y

    def fit(self, X_train, y_train):
        """Entrenar el pipeline siguiendo los pasos de PyCaret"""
        print("1. Imputando valores faltantes (estrategia: media)...")
        X_imputed = self.numerical_imputer.fit_transform(X_train)
        X_imputed = pd.DataFrame(X_imputed, columns=X_train.columns, index=X_train.index)

        print("2. Normalizando con MinMax...")
        X_normalized = self.scaler.fit_transform(X_imputed)
        X_normalized = pd.DataFrame(X_normalized, columns=X_train.columns, index=X_train.index)

        print("3. Entrenando CatBoostRegressor...")
        self.model.fit(X_normalized, y_train)

        self.is_fitted = True
        return self

    def transform(self, X):
        """Aplicar transformaciones del pipeline"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")

        # 1. Imputar
        X_imputed = self.numerical_imputer.transform(X)
        X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)

        # 2. Normalizar
        X_normalized = self.scaler.transform(X_imputed)
        X_normalized = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)

        return X_normalized

    def predict(self, X):
        """Realizar predicciones"""
        X_processed = self.transform(X)
        return self.model.predict(X_processed)

# PRIMERO: Configurar la página
st.set_page_config(
    page_title="Predicción de PCI", 
    layout="wide",
    page_icon="🛣️",
    initial_sidebar_state="expanded"
)

# Cargar modelo (SIN scaler)
@st.cache_resource
def load_model():
    with open("catboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_resource
def load_shap_explainer(_model):
    """Cargar el explainer de SHAP (cached para mejor performance)"""
    explainer = shap.TreeExplainer(_model.model)  # Acceder al modelo interno
    return explainer

model = load_model()
explainer = load_shap_explainer(model)

# Lista de variables del NUEVO modelo
feature_names = [
    'ini_pci',
    'edad_tupla',
    'TOTAL_PRECIP_WAVG',
    'CDD_WAVG',
    'CWD_WAVG',
    'Rx1day_Mar',
    'Rx1day_Apr',
    'Rx1day_Oct',
    'Rx1day_Nov',
    'Rx5day_May',
    'Rx5day_Jun',
    'Rx5day_Jul',
    'Rx5day_Dec',
    'ID_WAVG',
    'Tx35_WAVG',
    'TXn_Apr',
    'DTR_Dec',
    'SN_VALUE',
    'ANL_KESAL_LTPP_LN_YR_WAVG',
    'AADT_ALL_VEHIC_WAVG',
    'AADT_TRUCK_COMBO_WAVG'
]

# Restricciones de validación (min, max) - None significa sin restricción
validation_rules = {
    'ini_pci': (0, 100),
    'edad_tupla': (0, None),
    'TOTAL_PRECIP_WAVG': (0, None),
    'CDD_WAVG': (0, None),
    'CWD_WAVG': (0, None),
    'Rx1day_Mar': (0, None),
    'Rx1day_Apr': (0, None),
    'Rx1day_Oct': (0, None),
    'Rx1day_Nov': (0, None),
    'Rx5day_May': (0, None),
    'Rx5day_Jun': (0, None),
    'Rx5day_Jul': (0, None),
    'Rx5day_Dec': (0, None),
    'ID_WAVG': (0, None),
    'Tx35_WAVG': (0, None),
    'TXn_Apr': (None, None),  # Puede ser negativa
    'DTR_Dec': (0, None),
    'SN_VALUE': (0, None),
    'ANL_KESAL_LTPP_LN_YR_WAVG': (0, None),
    'AADT_ALL_VEHIC_WAVG': (0, None),
    'AADT_TRUCK_COMBO_WAVG': (0, None)
}

def validate_data(data, is_dataframe=True):
    """
    Validar datos contra las restricciones definidas
    Returns: (is_valid, errors_list)
    """
    errors = []
    
    if is_dataframe:
        # Validación para DataFrame (archivos)
        for idx, row in data.iterrows():
            for feature in feature_names:
                if feature in row:
                    value = row[feature]
                    min_val, max_val = validation_rules[feature]
                    
                    # Verificar mínimo
                    if min_val is not None and value < min_val:
                        errors.append(f"Fila {idx + 2}, campo '{feature}': valor {value} es menor que el mínimo permitido ({min_val})")
                    
                    # Verificar máximo
                    if max_val is not None and value > max_val:
                        errors.append(f"Fila {idx + 2}, campo '{feature}': valor {value} excede el máximo permitido ({max_val})")
    else:
        # Validación para diccionario (entrada manual)
        for feature in feature_names:
            if feature in data:
                value = data[feature]
                min_val, max_val = validation_rules[feature]
                
                # Verificar mínimo
                if min_val is not None and value < min_val:
                    errors.append(f"• {feature}: valor {value} es menor que el mínimo permitido ({min_val})")
                
                # Verificar máximo
                if max_val is not None and value > max_val:
                    errors.append(f"• {feature}: valor {value} excede el máximo permitido ({max_val})")
    
    return len(errors) == 0, errors

# Función para crear gráfico SHAP waterfall
def create_shap_waterfall(shap_values, expected_value, feature_values_processed, feature_values_original, feature_names):
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
    
    # Crear labels con valores originales
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

def create_deterioration_curve(user_input, model, feature_names):
    """Crear curva de deterioro del pavimento - SIN scaler"""
    
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
        
        # Predecir usando el nuevo modelo (sin scaler)
        input_df = pd.DataFrame([datos_año])
        prediccion = model.predict(input_df[feature_names])[0]
        
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
                    
                    # VALIDAR DATOS
                    is_valid, validation_errors = validate_data(input_data, is_dataframe=True)
                    
                    if not is_valid:
                        # Mostrar errores de validación
                        st.error("❌ **No se pueden realizar las predicciones. Hay errores en los datos:**")
                        for error in validation_errors:
                            st.error(error)
                        st.info("💡 Corrige los valores fuera de rango y vuelve a cargar el archivo.")
                    else:
                        # Continuar con las predicciones si todo está bien
                        # Predecir usando el nuevo modelo
                        predictions = model.predict(input_data)
                        
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
                            
                            # Calcular SHAP values (usando los datos procesados internamente)
                            input_processed = model.transform(input_data)
                            shap_values = explainer.shap_values(input_processed)
                            expected_value = explainer.expected_value
                            
                            # Mostrar explicación para cada predicción
                            for i in range(len(predictions)):
                                with st.expander(f"📊 Explicación predicción #{i+1} (PCI: {predictions[i]:.2f})"):
                                    shap_fig = create_shap_waterfall(
                                        shap_values[i], 
                                        expected_value, 
                                        input_processed.iloc[i].values, 
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
                                        user_input_from_file, model, feature_names
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
    
    # Organizar campos en categorías (ACTUALIZADAS para el nuevo modelo)
    categories = {
        "🏗️ Datos del pavimento": ["ini_pci", "edad_tupla", "SN_VALUE"],
        "🌡️ Datos climáticos": [
            'TOTAL_PRECIP_WAVG', 'CDD_WAVG', 'CWD_WAVG', 'Rx1day_Mar', 'Rx1day_Apr', 
            'Rx1day_Oct', 'Rx1day_Nov', 'Rx5day_May', 'Rx5day_Jun', 'Rx5day_Jul', 
            'Rx5day_Dec', 'ID_WAVG', 'Tx35_WAVG', 'TXn_Apr', 'DTR_Dec'
        ],
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
                        # Obtener restricciones para la variable
                        min_val, max_val = validation_rules[feature]
                        
                        # Configurar número input SIN límites automáticos para evitar errores de tipo
                        st.number_input(
                            label=feature.replace('_', ' ').title(),
                            key=f"input_{feature}",
                            help=f"Ingresa el valor para {feature}",
                            disabled=prediction_active,
                            step=1.0 if feature in ['edad_tupla', 'ID_WAVG', 'Tx35_WAVG'] else 0.1
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
            
            # VALIDAR DATOS ANTES DE PREDECIR
            is_valid, validation_errors = validate_data(user_input, is_dataframe=False)
            
            if not is_valid:
                # Mostrar errores de validación
                st.error("❌ **No se puede realizar la predicción. Corrige estos errores:**")
                for error in validation_errors:
                    st.error(error)
                st.info("💡 Ajusta los valores fuera de rango y vuelve a intentar.")
            else:
                # Proceder con la predicción si todo está bien
                # Guardar datos en session_state
                st.session_state['prediction_made'] = True
                st.session_state['user_input_data'] = user_input
                
                # Convertir a DataFrame
                input_df = pd.DataFrame([user_input])
                
                # Predecir usando el nuevo modelo (sin scaler)
                prediction = model.predict(input_df[feature_names])[0]
                
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
        
        # Usar el modelo para procesar los datos (sin scaler externo)
        input_processed = model.transform(input_df[feature_names])
        
        # Calcular SHAP values para la predicción individual
        shap_values = explainer.shap_values(input_processed)
        expected_value = explainer.expected_value
        
        # Crear gráfico SHAP waterfall
        shap_fig = create_shap_waterfall(
            shap_values[0], 
            expected_value, 
            input_processed.iloc[0].values, 
            list(user_input.values()),  # Valores originales
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
        deterioration_fig = create_deterioration_curve(user_input, model, feature_names)
        st.plotly_chart(deterioration_fig, use_container_width=True)

        # Detalles técnicos
        with st.expander("🔍 Ver detalles técnicos"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Datos procesados por el modelo:**")
                processed_df = pd.DataFrame(input_processed, columns=feature_names)
                st.dataframe(processed_df.T, use_container_width=True)
            with col2:
                st.markdown("**Datos originales:**")
                st.dataframe(input_df[feature_names].T, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🔬 Desarrollado con Streamlit y CatBoost | 📊 Modelo de Machine Learning para predicción de PCI</p>
</div>
""", unsafe_allow_html=True)