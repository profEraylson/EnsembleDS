import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from ModeloTeste import ModeloTeste


def select_k_similar_windows(training_windows, test_window, k): 
    
    distances= np.linalg.norm(training_windows - test_window, axis=1)
    k_similar_indices = np.argsort(distances)[:k]

    return k_similar_indices



def h_aware_selection_flexible(x_training_windows, y_training, test_window, k, models, h):
    """
    Seleção dinâmica sensível ao horizonte (h-aware), adaptada para reduzir h
    caso não existam janelas com todos os H passos futuros.

    Parâmetros:
        x_training_windows: np.ndarray (n_samples, n_lags)
        y_training: np.ndarray (n_samples,)
        test_window: np.ndarray (n_lags,)
        k: int - número de janelas mais próximas (RoC)
        models: list - modelos treinados com .predict()
        h: int - horizonte inicial da previsão

    Retorna:
        forecast: list - predições recursivas de t+1 a t+h'
        best_model_indices: list - índice do modelo selecionado para cada passo
        current_h: int - valor final de h efetivamente utilizado
    """

    # 1. Encontrar k janelas mais similares
    distances = np.linalg.norm(x_training_windows - test_window, axis=1)
    k_similar_indices = np.argsort(distances)[:k]
    X_roc = x_training_windows[k_similar_indices]

    # 2. Tentar reduzir h até conseguir pelo menos uma RoC válida
    current_h = h
    while current_h > 0:
        Y_roc_targets = []
        valid_roc_indices = []
        for idx in k_similar_indices:
            future = []
            for step in range(current_h):
                future_idx = idx + step
                if future_idx < len(y_training):
                    future.append(y_training[future_idx])
                else:
                    break
            if len(future) == current_h:
                Y_roc_targets.append(future)
                valid_roc_indices.append(idx)

        if len(Y_roc_targets) > 0:
            break  # Sucesso: pelo menos uma janela com h válidos encontrada
        current_h -= 1  # Reduz o horizonte

    if current_h == 0:
        raise ValueError("Não foi possível encontrar nenhuma RoC válida com pelo menos 1 passo futuro.")

    Y_roc_targets = np.array(Y_roc_targets)
    X_roc = x_training_windows[valid_roc_indices]

    # 3. Avaliar o erro recursivo por modelo e por horizonte
    n_models = len(models)
    errors_per_model_h = np.full((n_models, current_h), np.inf)

    for model_idx, model in enumerate(models):
        for i, window in enumerate(X_roc):
            window_copy = list(window)
            preds = []
            for step in range(current_h):
                x_input = np.array(window_copy).reshape(1, -1)
                pred = model.predict(x_input)[0]
                preds.append(pred)
                window_copy.append(pred)
                window_copy.pop(0)
            true = Y_roc_targets[i]
            errors_per_model_h[model_idx] += np.abs(np.array(preds) - true)

    errors_per_model_h /= len(Y_roc_targets)

    # 4. Selecionar o melhor modelo para cada passo h
    best_model_indices = np.argmin(errors_per_model_h, axis=0)

    # 5. Fazer a previsão recursiva com os modelos selecionados
    forecast = []
    current_window = list(test_window)
    for step in range(current_h):
        model = models[best_model_indices[step]]
        x_input = np.array(current_window).reshape(1, -1)
        pred = model.predict(x_input)[0]
        forecast.append(pred)
        current_window.append(pred)
        current_window.pop(0)

    return forecast, best_model_indices.tolist(), current_h





def h_aware_selection(x_training_windows, y_training, test_window, k, models, h):
    """
    Seleção dinâmica sensível ao horizonte (h-aware), onde para cada passo h,
    é selecionado o modelo que apresenta menor erro médio recursivo nos k padrões mais similares (RoC).

    Parâmetros:
        x_training_windows: np.ndarray, shape (n_samples, n_lags)
        y_training: np.ndarray, shape (n_samples,)
        test_window: np.ndarray, shape (n_lags,)
        k: int - número de janelas na RoC
        models: list - modelos já treinados com método .predict()
        h: int - horizonte da previsão multi-step

    Retorna:
        forecasts: list - previsões para t+1 até t+H
        selected_model_indices: list - índice do modelo selecionado para cada passo h
    """

    # Passo 1: Encontrar índices das k janelas mais similares à test_window
    k_similar_indices = select_k_similar_windows(x_training_windows, test_window, k)
    X_roc = x_training_windows[k_similar_indices]

    # Passo 2: Extrair os targets multi-step da RoC
    Y_roc_targets = []
    valid_roc_indices = []
    for idx in k_similar_indices:
        future = []
        for step in range(h):
            future_idx = idx + step
            if future_idx < len(y_training):
                future.append(y_training[future_idx])
            else:
                break
        
        if len(future) == h:
            Y_roc_targets.append(future)
            valid_roc_indices.append(idx)
    
    if len(Y_roc_targets) == 0:
        raise ValueError("Não foi possível extrair H passos futuros completos da RoC.")
    
    Y_roc_targets = np.array(Y_roc_targets)
    X_roc = x_training_windows[valid_roc_indices]
    
    # Passo 3: Avaliar erro por modelo e por horizonte h
    n_models = len(models)
    #errors_per_model_h = np.full((n_models, h), np.nan)
    errors_per_model_h = np.zeros((n_models, h))
    for model_idx, model in enumerate(models):
        for i, window in enumerate(X_roc):
            window_copy = list(window)
            preds = []
            for step in range(h):
                x_input = np.array(window_copy).reshape(1, -1)
                pred = model.predict(x_input)[0]
                preds.append(pred)
                window_copy.append(pred)
                window_copy.pop(0)
            true = Y_roc_targets[i]
            
            errors_per_model_h[model_idx, :] = np.abs(np.array(preds) - true)
    
    
    
    errors_per_model_h /= len(Y_roc_targets) #media para fazer o MAE
    
    # Passo 4: Selecionar melhor modelo para cada h
    best_model_indices = np.argmin(errors_per_model_h, axis=0)

    # Passo 5: Previsão recursiva usando o modelo escolhido para cada h
    forecast = []
    current_window = list(test_window)
    for step in range(h):
        model = models[best_model_indices[step]]
        x_input = np.array(current_window).reshape(1, -1)
        pred = model.predict(x_input)[0]
        forecast.append(pred)
        current_window.append(pred)
        current_window.pop(0)

    return forecast, best_model_indices.tolist()





models = [ModeloTeste(bias) for bias in [0.1, 0.5, 1]]

np.random.seed(42)
serie = np.arange(100) + np.random.normal(0, 2, 100)

# Parâmetros para janelas
window_size = 5
h = 3

# Gerar X (janelas) e Y (targets) para previsão recursiva
x_training_windows = []
y_training = []

for i in range(len(serie) - window_size):
    x_training_windows.append(serie[i:i+window_size])
    y_training.append(serie[i+window_size])  # target t+1

x_training_windows = np.array(x_training_windows)
y_training = np.array(y_training)

# Última janela observada
last_window = serie[-window_size:]


forecast, selected_indices = h_aware_selection(x_training_windows, y_training, last_window, k=10, models=models, h=5)
breakpoint()
