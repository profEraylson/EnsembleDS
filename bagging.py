import numpy as np
import pandas as pd
from preprocessamento import split_train_val_test, TimeSeriesNormalizer, select_lag_acf, create_windows_targets
from models import train_linear_regression

class BaggingEnsemble:

    def __init__(self, base_model_fn, n_models=10, val_pct=0.32, combine_method='mean'):
        """
        Parâmetros:
        - base_model_fn: função que recebe (x_train, y_train, x_val, y_val) e retorna (modelo, _, modelo_selecionado)
        - n_models: número de modelos no ensemble
        - val_pct: percentual dos dados usados para validação
        - combine_method: 'mean' ou 'median' para combinar previsões
        """
        assert combine_method in ['mean', 'median'], "combine_method deve ser 'mean' ou 'median'"
        
        self.base_model_fn = base_model_fn
        self.n_models = n_models
        self.val_pct = val_pct
        self.combine_method = combine_method
        self.models = []
        self.models_val_performance = [] 
        self.indices = []
        self.train_indices = []
        self.last_selected_model = None
        

    def _bootstrap_indices(self, data_len):
        return np.random.randint(0, data_len, data_len)

    
    def fit(self, X, y):
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        full_data = np.hstack([X, y])
        
        for i in range(self.n_models):
            print(f"Training model {i+1}/{self.n_models}")

            indices = self._bootstrap_indices(len(full_data))
            partition = full_data[indices, :]

            X_part, y_part = partition[:, :-1], partition[:, -1]
            val_size = int(len(y_part) * self.val_pct)

            x_train = X_part[:-val_size]
            y_train = y_part[:-val_size]
            x_val = X_part[-val_size:]
            y_val = y_part[-val_size:]

            train_idx = indices[:-val_size]

            model, performance_val = self.base_model_fn(x_train, y_train, x_val, y_val)

            self.models.append(model)
            self.indices.append(indices)
            self.train_indices.append(train_idx)
            self.models_val_performance.append(performance_val)
            

    def predict(self, X_test):
        """
        Faz a previsão sobre uma ou mais janelas de teste.

        Parâmetros:
        - X_test: array 2D de shape (n_janelas, n_features) ou array 1D (1 janela)

        Retorna:
        - y_pred: array de previsões agregadas (n_janelas,)
        """
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        all_preds = []

        for model in self.models:
            try:
                pred = model.predict(X_test)
            except:
                pred = model.predict(X_test.astype(np.float32))  # fallback para deep models
            all_preds.append(pred)

        all_preds = np.array(all_preds)  # shape: (n_models, n_janelas)

        if self.combine_method == 'mean':
            return np.mean(all_preds, axis=0)
        else:  # median
            return np.median(all_preds, axis=0)

    
    def predict_recursive(self, last_window, h):
        """
        Faz previsão recursiva multi-step ahead.

        Parâmetros:
        - last_window: array 1D com os últimos valores (tamanho deve cobrir os lags necessários)
        - h: número de passos à frente

        Retorna:
        - y_preds: array com h previsões
        """
     
        window = list(last_window)
        y_preds = []

        for step in range(h):
            
            
            input_array = np.array(window).reshape(1, -1)
            y_pred = self.predict(input_array)[0]
            y_preds.append(y_pred)
            
            # Atualiza a janela descartando o primeiro e inserindo a nova previsão no fim
            window = window[1:] + [y_pred]

        return np.array(y_preds)
    
    
    def get_ensemble(self):
        return {
            'models': self.models,
            'indices': self.indices,
            'indices_train': self.train_indices
        }

    def get_selected_model(self):
        return self.last_selected_model


if __name__ == '__main__':


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
 

    ensemble = BaggingEnsemble(
        base_model_fn=train_linear_regression,
        n_models=50,
        val_pct=0.2,    
        combine_method='median'
        )

    ensemble.fit(X_train, y_train)
    
    prev = ensemble.predict(X_test)
    
    prev_horiz = ensemble.predict_recursive(X_test[0, :], h=y_test.shape[0])
    

    






#     np.random.seed(42)
#     print('Série:', serie_name)

    
#     p_tr = 0.75 #75% treinamento
 
#     train, test = split_train_val_test(serie_normalizada, p_tr)
#     serie_normalizada = normalise(serie)

#     max_lag = 20
#     lags_acf = select_lag_acf(serie_normalizada, max_lag)
#     max_sel_lag = lags_acf[0]
#     train_lags = create_windows(train, max_sel_lag+1)
#     test_lags = create_windows(test, max_sel_lag+1) #Utilizar as 20 instancia iniciais para gerar o janelamento

#     train_val = train_lags[ : -(len(test_lags))] # treinamento antes de fazer a validação, ai nao treina com todo o conjunto de treinamento deixando o conjunto de validação de fora desse primeiro treinamento
#     val_lags = train_lags[-(len(test_lags)): ]

#     '''-------------------------------------------------'''

#     #MODELOS GERADOS COM 50% DO CONJUNTO DE TREINAMENTO, PARA PREVER O CONJUTNO DE VALIDAÇÃO bagging
#     print("BASE 50%")
#     X_train_val, y_train_val = train_val[:, 0:-1], train_val[:, -1]
#     ensemble, msb = bagging(int(quantidade_modelo), X_train_val, y_train_val, lags_acf)

#     ensemble_condig = {'ensemble': ensemble['models'], 'acf': lags_acf, 'indices_train': ensemble['indices_train'], 'indices_val_train': ensemble['indices']}
#     nome_arquivo = 'ensemble_Train\\'+serie_name+'_'+msb+'_pool.pkl'
#     #nome_arquivo = 'ensemble_Train/'+serie_name+'_'+msb+'_pool.pkl' #CLUSTER

#     pickle.dump( ensemble_condig, open( nome_arquivo, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

#     # ## USAR COM ELM
#     # import dill
#     # dill.dump(ensemble_condig, open(nome_arquivo, "wb"), protocol=pickle.HIGHEST_PROTOCOL)

#     '''-------------------------------------------------'''

#     #TREINAR NA BASE COMPLETA DE TREINAMENTO bagging
#     print("BASE 75%")
#     X_train, y_train = train_lags[:, 0:-1], train_lags[:, -1]
#     ensemble, msb = bagging(int(quantidade_modelo), X_train, y_train, lags_acf)

#     x_test, y_test = test_lags[:, 0:-1], test_lags[:, -1]
#     desempMed = desempenho_media_pool(ensemble['models'], x_test[:, lags_acf], y_test) 
#     ensemble_condig = {'ensemble': ensemble['models'], 'acf': lags_acf, 'indices_train': ensemble['indices_train'], 'indices_val_train': ensemble['indices']}
#     nome_arquivo = 'ensemble_TrainVal\\'+serie_name+'_'+msb+'_pool.pkl'
#     #nome_arquivo = 'ensemble/'+serie_name+'_'+msb+'_pool.pkl' #CLUSTER
#     pickle.dump( ensemble_condig, open( nome_arquivo, "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

#     # ## USAR COM ELM
#     # import dill
#     # dill.dump(ensemble_condig, open(nome_arquivo, "wb"), protocol=pickle.HIGHEST_PROTOCOL)



#     # caminho_arquivo = f"pool_desempMed_{serie_name}.txt"
#     # with open(caminho_arquivo, "w") as arquivo:
#     #     arquivo.write(str(desempMed))



# if __name__ == "__main__": #python bagging.py airline

#     ## creating the arg's parser
#     parser = argparse.ArgumentParser(description='Select the dataset')
#     parser.add_argument('dataset', metavar='P', type=str, help='datasets: airline, amazon, apple, carsales, eletric, gas' +
#                         'goldman, lake, microsoft, nordic, pigs, pollutions, star, sunspot, vehicle, traffic, wine') # metavar='P' significa que, quando alguém executar o script e solicitar ajuda (--help), o argumento dataset será exibido na mensagem de ajuda como P
    
#     parser.add_argument('quantidade', metavar='Q', type=str, help='quantidades: ')
    
#     # parse the args
#     args = parser.parse_args()
#     tic = time.time()
    
#     # execution of the main function
#     print(f'[INFO]: STARTING | {args.dataset}')
#     main(args.dataset, args.quantidade)
#     toc = time.time()
#     print(f'[INFO]: {args.dataset} | Execution time: {toc-tic}\n')














