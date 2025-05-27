import numpy as np
from sklearn.metrics import mean_absolute_error as MAE
from ModeloTeste import ModeloTeste
import pandas as pd
from preprocessamento import split_train_val_test, TimeSeriesNormalizer, select_lag_acf, create_windows_targets
from models import train_linear_regression
from bagging import BaggingEnsemble
import matplotlib.pyplot as plt


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
    plt.plot(y_test, label = 'y')
    plt.plot(preds_h_prev, label = 'h_prev')
    plt.plot(preds_prox, label = 'prox')
    plt.plot(preds_roc, label = 'roc')
    plt.legend()
    plt.show()
    breakpoint()

    
    



   


