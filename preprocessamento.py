import numpy as np
np.random.seed(42)

from sklearn.preprocessing import MinMaxScaler
import numpy as np

class TimeSeriesNormalizer:

    def __init__(self, scaler=None):
        """
        Inicializa o normalizador com o scaler fornecido. Por padrão, usa MinMaxScaler.

        Parâmetros:
        - scaler: instância de um scaler do sklearn, como MinMaxScaler, StandardScaler, etc.
        """
        self.scaler = scaler if scaler is not None else MinMaxScaler()
        self.fitted = False

    def fit(self, series):
        """
        Ajusta o scaler aos dados da série temporal.

        Parâmetros:
        - series: array-like (1D ou 2D). Se for 1D, será convertido para 2D.
        """
        series = self._ensure_2d(series) #exigencia do sklearn 
        self.scaler.fit(series)
        self.fitted = True

    def transform(self, series):
        """
        Aplica a transformação de normalização à série.

        Parâmetros:
        - series: array-like (1D ou 2D). Se for 1D, será convertido para 2D.

        Retorna:
        - Série normalizada (mesmo formato da entrada).
        """
        if not self.fitted:
            raise RuntimeError("O método 'fit' deve ser chamado antes de 'predict'.")

        original_shape = np.shape(series)
        series = self._ensure_2d(series)
        normalized = self.scaler.transform(series)

        if len(original_shape) == 1:
            return normalized.flatten()
        return normalized
    
    def inverse_transform(self, series_normalized):
        ## Não testei!!
        """
        Reverte a normalização para a escala original.
        """
        if not self.fitted:
            raise RuntimeError("O método 'fit' deve ser chamado antes de 'inverse_transform'.")

        original_shape = np.shape(series_normalized)
        series_normalized = self._ensure_2d(series_normalized)
        restored = self.scaler.inverse_transform(series_normalized)

        if len(original_shape) == 1:
            return restored.flatten()
        return restored

    def _ensure_2d(self, data):
        """
        Garante que os dados estejam em formato 2D (necessário para scalers do sklearn).
        """
        data = np.array(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data







def normalise_interval(minimo, maximo, serie):
	from sklearn.preprocessing import MinMaxScaler
	scaler = MinMaxScaler(feature_range=(minimo, maximo))
	scaler = scaler.fit(serie)
	normalized = scaler.transform(serie)
	return normalized, scaler 
	
	
	
def desnorm_interval(serie_norm, serie_real, minimo, maximo):
	norm, scaler = normalise_interval(minimo, maximo, serie_real)
	inversed = scaler.inverse_transform(serie_norm)
	return inversed

def split_serie_with_lags(serie, perc_train, perc_val = 0):
    
    #faz corte na serie com as janelas já formadas 
    
    x_date = serie[:, 0:-1]
    y_date = serie[:, -1]        
       
    train_size = np.fix(len(serie) *perc_train)
    train_size = train_size.astype(int)
    
    if perc_val > 0:        
        val_size = np.fix(len(serie) *perc_val).astype(int)
              
        
        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]
        print("Particao de Treinamento:", 0, train_size  )
        
        x_val = x_date[train_size:train_size+val_size,:]
        y_val = y_date[train_size:train_size+val_size]
        
        print("Particao de Validacao:",train_size, train_size+val_size)
        
        x_test = x_date[(train_size+val_size):,:]
        y_test = y_date[(train_size+val_size):]
        
        print("Particao de Teste:", train_size+val_size, len(y_date))
        
        return x_train, y_train, x_test, y_test, x_val, y_val
        
    else:
        
        x_train = x_date[0:train_size,:]
        y_train = y_date[0:train_size]

        x_test = x_date[train_size:-1,:]
        y_test = y_date[train_size:-1]

        return x_train, y_train, x_test, y_test




def create_windows_targets(serie, size_window, h):
    """
    Gera X e Y para previsão de séries temporais com janelas deslizantes.
    
    Parâmetros:
    - serie (array-like): Série temporal original.
    - janela (int): Tamanho da janela de entrada.
    - h (int): Passos à frente para prever (horizon).
    
    Retorna:
    - X (np.ndarray): Matriz de janelas (n_amostras, janela).
    - Y (np.ndarray): Vetor de targets (n_amostras,).
    """
    serie = np.array(serie)
    X, Y = [], []
    
    for i in range(len(serie) - size_window - h + 1):
        x_i = serie[i:i+size_window]
        y_i = serie[i+size_window+h-1]  # valor h passos à frente
        X.append(x_i)
        Y.append(y_i)

    return np.array(X), np.array(Y)



def select_lag_acf(serie, max_lag):  #pacf
    from statsmodels.tsa.stattools import acf #A função acf que você mencionou é parte do módulo statsmodels.tsa.stattools da biblioteca Statsmodels em Python e é usada para calcular a função de autocorrelação (ACF) de uma série temporal. A função de autocorrelação é uma medida estatística que indica o grau de correlação entre uma série temporal e suas próprias versões passadas em vários lags (atrasos temporais). Em resumo, a função ACF percorre a série temporal com o intervalo especificado de lags para calcular a correlação entre a série original e suas versões defasadas, ajudando na identificação de padrões de correlação temporal.
    x = serie[0: max_lag+1] 

    
    acf_x, confint = acf(serie, nlags=max_lag, alpha=.05) #A função acf é útil para entender a autocorrelação na série temporal, o que pode ser importante em análise de séries temporais, modelagem e previsão. Através da ACF, você pode identificar padrões de sazonalidade e determinar a ordem de modelos autorregressivos (AR) em processos autorregressivos integrados de média móvel (ARIMA).
    
    
    limiar_superior = confint[:, 1] - acf_x 
    limiar_inferior = confint[:, 0] - acf_x

    lags_selecionados = []
    
    for i in range(1, max_lag+1):

        
        if acf_x[i] >= limiar_superior[i] or acf_x[i] <= limiar_inferior[i]:
            lags_selecionados.append(i-1)  #-1 por conta que o lag 1 em python é o 0
    
    #caso nenhum lag seja selecionado, essa atividade de seleção para o gridsearch encontrar a melhor combinação de lags
    if len(lags_selecionados)==0:


        print('NENHUM LAG POR ACF')
        lags_selecionados = [i for i in range(max_lag)]

    print('LAGS', lags_selecionados)
    #inverte o valor dos lags para usar na lista de dados
    lags_selecionados = [max_lag - (i+1) for i in lags_selecionados]



    return lags_selecionados

def split_serie_less_lags(series, perc_train, perc_val = 0): 
    import numpy as np   
      
    train_size = np.fix(len(series) *perc_train)
    train_size = train_size.astype(int)
    
    if perc_val > 0:
        
        val_size = np.fix(len(series) *perc_val).astype(int)
        
        x_train = series[0:train_size]
        x_val = series[train_size:train_size+val_size]        
        x_test = series[(train_size+val_size):-1]

        return x_train, x_test, x_val
        
    else:
        
                
        x_train = series[0:train_size+1]
        x_test = series[train_size:-1]
        

        return x_train, x_test
		
def select_validation_sample(serie, perc_val):
    tam = len(serie)
    val_size = np.fix(tam *perc_val).astype(int)
    return serie[0:tam-val_size,:],  serie[tam-val_size:-1,:]




def split_train_val_test(series, p_tr, perc_val = 0):

    tam_serie =  len(series)
    #print(tam_serie)
    train_size = int(np.ceil(p_tr * tam_serie))
    
    if perc_val > 0:
        
        val_size = int(np.ceil(len(series) *perc_val))
        
        
        
        x_train = series[0:train_size]
        x_val = series[train_size:train_size+val_size]        
        x_test = series[(train_size+val_size):]
        
        return x_train, x_test, x_val
        
    else:
        
                
        x_train = series[0:train_size]
        x_test = series[train_size:]
        

        return x_train, x_test