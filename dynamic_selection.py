import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from ModeloTeste import ModeloTeste
import pandas as pd
from preprocessamento import split_train_val_test, TimeSeriesNormalizer, select_lag_acf, create_windows_targets
from models import train_linear_regression
from bagging import BaggingEnsemble
import matplotlib.pyplot as plt
from metrics import evaluate_models

def select_k_similar_windows(training_windows, test_window, k): 
    
    distances= np.linalg.norm(training_windows - test_window, axis=1)
    k_similar_indices = np.argsort(distances)[:k]

    return k_similar_indices



def ola_ds_selection(x_training_windows, y_training, test_window, k, n, models):

    #RoC
    k_similar_indices = select_k_similar_windows(x_training_windows, test_window, k)
    X_roc = x_training_windows[k_similar_indices]
    y_roc = y_training[k_similar_indices]

    #Selection
    erros_models =  []
    for model in models:
        pred = model.predict(X_roc)
        
        mae = float(MAE(pred, y_roc))
        erros_models.append((model, mae))
    
    
    selected_models_erros = sorted(erros_models, key=lambda x: x[1])[:n]
    selected_models = [model for model, _ in selected_models_erros]
    selected_maes = [mae for _, mae in selected_models_erros]
    return selected_models, selected_maes


def proxy_ds(x_training_windows, y_training, last_window, k, n, models, h):
    
    '''1ª Ideia: seleciona os h-step ahead models utilizando a previsão t como um substituto do valor real'''
    preds = []
    for _ in range(h):
        
        selected_models, _ = ola_ds_selection(x_training_windows, y_training, last_window, k, n, models)
        prev = selected_models[0].predict(last_window.reshape(1, -1))
        preds.append(prev.item())        
        last_window = np.roll(last_window, -1)  
        last_window[-1] = prev.item()

    return preds



def select_best_model(training_windows, target_values, h, models):
    '''Utilizado para a abordagem que seleciona o melhor modelo com base nas ultimas janelas
      e utiliza o mesmo modelo para prever todo h-step ahead'''

    # Selecionar as h últimas janelas de treinamento
    last_h_windows = training_windows[-h:]
    last_h_targets = target_values[-h:]

    mae_scores = []

    # Avaliar cada modelo
    for model in models:
        # Fazer a previsão usando as últimas h janelas
        predictions = model.predict(last_h_windows)

        # Calcular o MAE para o modelo
        mae = MAE(last_h_targets, predictions)
        mae_scores.append(mae)

    # Identificar o índice do modelo com o menor MAE
    best_model_index = np.argmin(mae_scores)
    print(f"Melhor modelo pelo ds_h_previous_roc: {best_model_index} ")
    best_model = models[best_model_index]
    

    return best_model, mae_scores


def recursive_forecast(model, last_window, h):
    #Faz a previsão h-step ahead utilizando abordagem recursiva     

    predictions = []
    current_input = last_window.copy()

    for _ in range(h):
        # Realizar a previsão para o próximo passo
        
        next_prediction = model.predict(current_input.reshape(1, -1))[0]
        
        # Adicionar a previsão à lista de previsões
        predictions.append(float(next_prediction))

        # Atualizar a entrada removendo o valor mais antigo e adicionando a nova previsão
        current_input = np.roll(current_input, -1)  
        current_input[-1] = next_prediction
    

    return predictions


def ds_h_previous(training_windows, target_values, last_window, h, models):
    
    '''
    Ideia semelhante ao hold-out (base line)
    Seleciona o melhor modelo nas ultimas h janelas e ele faz previsão dos h pontos'''
    model = select_best_model(training_windows, target_values, h, models)[0]
    prevs = recursive_forecast(model, last_window, h)
    return prevs



def ds_h_previous_roc(training_windows, training_targets, test_window, h, models, k):
    '''2ª IDEIA:
    Seleciona o modelo que apresenta maior desempenho em prever as h step de cada janela da ROC 
    Forma recursiva. Modelo selecionado faz a previsão multi-step
    '''

     
    # Calcula a similaridade (distância euclidiana) entre a janela de teste e as janelas de treinamento
    distances = np.linalg.norm(training_windows - test_window, axis=1)

    # Seleciona os índices das k janelas mais similares
    k_indices = np.argsort(distances)[:k]

    
    model_maes = []

    # Itera sobre os modelos do Pool
    for model in models:
        total_mae = 0
        valid_janelas = 0

        # Buscar por janelas similares mas que tem h valores a frente
        for idx in k_indices:
            next_indices = range(idx + 1, idx + 1 + h)
            
            # Verifica se algum índice ultrapassa o tamanho de training_windows
            valid_indices = [i for i in next_indices if i < len(training_windows)]
            
            if valid_indices:
                next_windows = training_windows[valid_indices]
                next_targets = training_targets[valid_indices]

                # Faz as previsões para as janelas válidas
                #predictions = [model.predict(w.reshape(1, -1))[0] for w in next_windows]
                predictions = model.predict(next_windows)

                # Calcula o MAE para essas previsões
                total_mae += MAE(next_targets, predictions)
                valid_janelas += 1

        # Calcula o MAE médio para o modelo
        avg_mae = total_mae / valid_janelas if valid_janelas > 0 else float('inf')
        model_maes.append(avg_mae)
    
    
    # Seleciona o índice do modelo com menor MAE
    best_model_index = np.argmin(model_maes)
    model = models[best_model_index]
    print(f"Melhor modelo pelo ds_h_roc: {best_model_index} ")
    prevs = recursive_forecast(model, test_window, h)
    
    return prevs




def h_aware_selection_flexible(training_windows, training_targets, test_window, k, models, h):
    """
    Seleção dinâmica sensível ao horizonte (h-aware), com fallback parcial ou total via média das previsões.

    Parâmetros:
        training_windows: np.ndarray (n_samples, n_lags)
        training_targets: np.ndarray (n_samples,)
        test_window: np.ndarray (n_lags,)
        k: int - número de janelas mais próximas (RoC)
        models: list - modelos treinados com .predict()
        h: int - horizonte total da previsão

    Retorna:
        forecast: list - predições recursivas de t+1 a t+h
        best_model_indices: list or None - índices dos modelos selecionados nos primeiros h_roc passos (se houver)
        current_h: int - valor h_roc efetivo utilizado na RoC (pode ser 0 até h)
    """

    # 1. Encontrar k janelas mais similares
    distances = np.linalg.norm(training_windows - test_window, axis=1)
    k_similar_indices = np.argsort(distances)[:k]

    # 2. Tentar construir RoC com horizonte reduzido se necessário
    current_h = h
    while current_h > 0:
        Y_roc_targets = []
        valid_roc_indices = []
        for idx in k_similar_indices:
            future = []
            for step in range(current_h):
                future_idx = idx + step
                if future_idx < len(training_targets):
                    future.append(training_targets[future_idx])
                else:
                    break
            if len(future) == current_h:
                Y_roc_targets.append(future)
                valid_roc_indices.append(idx)

        if len(Y_roc_targets) > 0:
            break
        current_h -= 1  # reduz h_roc

    # Caso 1: nenhum passo disponível para RoC
    if current_h == 0:
        print("Nenhuma RoC válida encontrada. Usando fallback por média para todos os h passos.")
        forecast = []
        current_window = list(test_window)
        for step in range(h):
            preds = []
            for model in models:
                x_input = np.array(current_window).reshape(1, -1)
                pred = model.predict(x_input)[0]
                preds.append(pred)
            pred_mean = np.mean(preds)
            forecast.append(pred_mean)
            current_window.append(pred_mean)
            current_window.pop(0)
        return forecast

    else:
        print(f"RoC construída com {current_h} passos à frente")
    
    # 3. Construir RoC com current_h
    Y_roc_targets = np.array(Y_roc_targets)
    X_roc = training_windows[valid_roc_indices]
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

    # 4. Selecionar o melhor modelo para cada passo nos h_roc
    best_model_indices = np.argmin(errors_per_model_h, axis=0)

    # 5. Previsão recursiva: RoC + fallback para o restante
    forecast = []
    current_window = list(test_window)

    for step in range(h):
        if step < current_h:
            model = models[best_model_indices[step]]
            x_input = np.array(current_window).reshape(1, -1)
            pred = model.predict(x_input)[0]
        else:
            preds = []
            for model in models:
                x_input = np.array(current_window).reshape(1, -1)
                pred_model = model.predict(x_input)[0]
                preds.append(pred_model)
            pred = np.mean(preds)  # fallback por média

        forecast.append(pred)
        current_window.append(pred)
        current_window.pop(0)

    return forecast 


def h_aware_selection(training_windows, training_targets, test_window, k, models, h):
    """
    Seleção dinâmica sensível ao horizonte (h-aware), onde para cada passo h,
    é selecionado o modelo que apresenta menor erro médio recursivo nos k padrões mais similares (RoC).

    Parâmetros:
        training_windows: np.ndarray, shape (n_samples, n_lags)
        training_targets: np.ndarray, shape (n_samples,)
        test_window: np.ndarray, shape (n_lags,)
        k: int - número de janelas na RoC
        models: list - modelos já treinados com método .predict()
        h: int - horizonte da previsão multi-step

    Retorna:
        forecasts: list - previsões para t+1 até t+H
        selected_model_indices: list - índice do modelo selecionado para cada passo h
    """

    # Passo 1: Encontrar índices das k janelas mais similares à test_window
    k_similar_indices = select_k_similar_windows(training_windows, test_window, k)
    X_roc = training_windows[k_similar_indices]

    # Passo 2: Extrair os targets multi-step da RoC
    Y_roc_targets = []
    valid_roc_indices = []
    for idx in k_similar_indices:
        future = []
        for step in range(h):
            future_idx = idx + step
            if future_idx < len(training_targets):
                future.append(training_targets[future_idx])
            else:
                break
        
        if len(future) == h:
            Y_roc_targets.append(future)
            valid_roc_indices.append(idx)
    
    if len(Y_roc_targets) == 0:
        raise ValueError("Não foi possível extrair H passos futuros completos da RoC.")
    
    Y_roc_targets = np.array(Y_roc_targets)
    X_roc = training_windows[valid_roc_indices]
    
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




if __name__ == "__main__":

    path = 'https://raw.githubusercontent.com/EraylsonGaldino/dataset_time_series/master/airline.txt'
    df_serie = pd.read_csv(path, header=None)
    df_serie['date'] = pd.date_range(start='1950', periods=df_serie.shape[0], freq='ME')
    serie = df_serie[0]
    p_tr = 0.75 #75% treinamento
    train, test = split_train_val_test(serie, p_tr)

    normalizer = TimeSeriesNormalizer()
    normalizer.fit(train)
    train_norm = normalizer.transform(train)
    test_norm = normalizer.transform(test)
    max_lag = 20
    lags_acf = select_lag_acf(train_norm, max_lag)
    max_sel_lag = lags_acf[0]
    X_train, y_train = create_windows_targets(train_norm, max_sel_lag, h=1)
    X_test, y_test = create_windows_targets(test_norm, max_sel_lag,  h=1) 

    k = 2

    ensemble = BaggingEnsemble(
        base_model_fn=train_linear_regression,
        n_models=50,
        val_pct=0.2,    
        combine_method='median'
        )

    ensemble.fit(X_train, y_train)
   

    n = 1
    h = 17
    test_window = X_test[0, :]
    models = ensemble.get_ensemble()['models']
    preds_prox = proxy_ds(X_train, y_train, test_window, k, n, models, h)
    preds_h_prev = ds_h_previous(X_train, y_train, test_window, h, models)
    preds_roc = ds_h_previous_roc(X_train, y_train, test_window, h, models, k)
    preds_aware = h_aware_selection_flexible(X_train, y_train, test_window, k, models, h)
    
    plt.plot(y_test, label = 'y')
    plt.plot(preds_h_prev, label = 'h_prev')
    plt.plot(preds_prox, label = 'prox')
    plt.plot(preds_roc, label = 'roc')
    plt.plot(preds_aware, label = 'aware')
    plt.legend()
    plt.show()
    # breakpoint()

    models_prevs = [
        preds_prox, 
        preds_aware,
        preds_h_prev,
        preds_roc
    ]

    models_name = [
        'prox',
        'aware',
        'h_prev',
        'roc'
    ]
    df_resultados = evaluate_models(models_prevs, models_name, y_test)
    breakpoint()

   

    
    



   


