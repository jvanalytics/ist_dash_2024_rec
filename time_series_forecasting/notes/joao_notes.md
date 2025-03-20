# Exame pst


A. Características de séries temporais
### a) A série Original representada na Figure 1 apresenta uma tendência (trend). VERDADEIRO
- Ver linha da media ao longo da serie.
### b) A série Original representada na Figure 1 apresenta uma clara sazonalidade.FALSO
- Nao se vê repetiçao?
### c) Todas as transformações apresentadas na Figure 2 resultam da aplicação da uma transformação de agregação. FALSO
- Falso? como assim? a figura indica a transformaçao
### d) As transformações T1 e T3 usam janelas temporais mais largas do que as transformações T2 e T4. FALSO
- T4 usa a janela mais larga!


B. Forecasting

a) Sabendo que a série Original tem uma granularidade de 10 minutos, se fosse treinada uma LSTM com
segmentos de um dia desta série, a rede teria 10 unidades LSTM (10 estados). FALSO
LSTM = 24 * 6 estados (6 x 10 minutos por hora) e nao 10 estados

b) Um modelo ARIMA (p=6, d=0, q=6) considera o mesmo número de instantes de tempo na previsão de cada
valor, que uma LSTM com 7 unidades LSTM (7 estados). VERDADEIRO

- A frase está correta. Um modelo ARIMA (p=6, d=0, q=6) e uma LSTM com 7 unidades LSTM (7 estados) consideram um número semelhante de instantes de tempo na previsão de cada valor.

    - Explicação:
    -   ARIMA (p=6, d=0, q=6):
        -   p (Auto-regressive part): O modelo usa os últimos 6 valores passados para prever o próximo valor.
        -   d (Differencing part): Como d=0, não há diferenciação aplicada.
        -   q (Moving average part): O modelo usa os últimos 6 erros passados para ajustar a previsão.
        -   Portanto, o modelo ARIMA está considerando 6 instantes de tempo passados para fazer a previsão.

    -   LSTM com 7 unidades:
        -   Uma LSTM (Long Short-Term Memory) é uma rede neural recorrente que pode aprender dependências de longo prazo.
        -   Com 7 unidades LSTM, a rede está configurada para considerar 7 estados anteriores (ou instantes de tempo) para fazer a previsão.
        -   Assim, ambos os modelos estão considerando um número semelhante de instantes de tempo (6 para ARIMA e 7 para LSTM) para fazer   -   previsões.


c) Um modelo ARIMA (p=6, d=0, q=0) aplicado para prever a série Original, teria um desempenho igual ao do modelo de rolling mean com w=6. FALSO

- A frase está incorreta. Um modelo ARIMA (p=6, d=0, q=0) e um modelo de rolling mean com w=6 não teriam o mesmo desempenho.

    - Explicação:
    -   ARIMA (p=6, d=0, q=0):
        -   p (Auto-regressive part): O modelo usa os últimos 6 valores passados para prever o próximo valor.
        -   d (Differencing part): Como d=0, não há diferenciação aplicada.
        -   q (Moving average part): Como q=0, não há média móvel dos erros passados.
        -   Portanto, o modelo ARIMA está considerando apenas os valores passados para fazer a previsão, sem suavização adicional.

    -   Rolling mean com w=6:
        -   Um modelo de rolling mean calcula a média dos últimos 6 valores (janela de tamanho 6) para prever o próximo valor.
        -   Este método suaviza a série temporal, reduzindo a variabilidade e ruído.

    -   Diferença de desempenho:
        -   O modelo ARIMA (p=6, d=0, q=0) pode capturar padrões mais complexos na série temporal devido à sua componente auto-regressiva.
        -   O modelo de rolling mean com w=6 é uma técnica de suavização simples que pode não capturar padrões complexos, resultando em previsões menos precisas.
        -   Portanto, os dois modelos não teriam o mesmo desempenho na previsão da série Original.

d) Se um modelo ARIMA (p=6, d=0, q=6) fosse aplicada para prever a série Original, seriam esperados bons
resultados. VERDADEIRO

- Explicação:

    - Um modelo ARIMA (p=6, d=0, q=6) é um modelo autoregressivo integrado de média móvel que utiliza 6 termos autoregressivos (p=6) e 6 termos de média móvel (q=6) para prever a série temporal. A componente d=0 indica que não há diferenciação aplicada à série.

    - Termos Autoregressivos (p=6): O modelo usa os últimos 6 valores passados da série temporal para prever o próximo valor. Isso permite que o modelo capture padrões e tendências de curto prazo na série.
    - Termos de Média Móvel (q=6): O modelo usa os últimos 6 erros passados (resíduos) para ajustar a previsão. Isso ajuda a capturar a variabilidade e o ruído na série temporal, melhorando a precisão das previsões.
    - A combinação de termos autoregressivos e de média móvel permite que o modelo ARIMA (p=6, d=0, q=6) capture tanto padrões de curto prazo quanto a variabilidade na série temporal, resultando em previsões mais precisas.

    - Portanto, ao aplicar um modelo ARIMA (p=6, d=0, q=6) para prever a série Original, seriam esperados bons resultados, desde que a série temporal não apresente características que exijam diferenciação (d>0) para se tornar estacionária.


C. Evaluation
Considere S1=[4.6, 4.9, 5, 4.7, 4.9, 5.1] e S2=[4.3, 5.1, 5, 4.9, 5, 5] como a sua previsão feita pelo modelo M, e R2
DECORAR FORMULA R2

a) O modelo rolling mean com w=1 teria um desempenho igual ao modelo de persistência. VERDADEIRO

- Explicação:

    - Modelo de Persistência: O modelo de persistência (ou "naive") prevê que o próximo valor na série será igual ao valor atual. Em outras palavras, ele simplesmente "persiste" o valor atual para a próxima previsão.

    - Modelo Rolling Mean com w=1: Um modelo de média móvel (rolling mean) com uma janela de tamanho 1 (w=1) calcula a média dos últimos 1 valor, que é o próprio valor atual. Portanto, a previsão para o próximo valor será igual ao valor atual.

    - Como ambos os modelos, de persistência e rolling mean com w=1, utilizam o valor atual da série para prever o próximo valor, eles terão o mesmo desempenho. Ambos os modelos farão previsões idênticas, resultando em métricas de desempenho iguais.

    - Portanto, o modelo rolling mean com w=1 teria um desempenho igual ao modelo de persistência.


b) O MAE para o modelo M seria menor do que o MAE depois de aplicar o modelo rolling mean com w=3.

Para responder a esta questão, precisamos calcular o MAE (Mean Absolute Error) para o modelo M e para o modelo rolling mean com w=3.

### Cálculo do MAE para o modelo M
O MAE é calculado como a média dos erros absolutos entre as previsões e os valores reais.

Vamos considerar S1 como os valores reais e S2 como as previsões feitas pelo modelo M.

\[ \text{MAE}_{M} = \frac{1}{n} \sum_{i=1}^{n} |S1_i - S2_i| \]

Onde:

\[ S1 = [4.6, 4.9, 5, 4.7, 4.9, 5.1] \]
\[ S2 = [4.3, 5.1, 5, 4.9, 5, 5] \]

### Cálculo do MAE para o modelo rolling mean com w=3
Para o modelo rolling mean com w=3, precisamos calcular a média móvel dos últimos 3 valores de S1 e comparar com os valores reais de S1.

\[ \text{Rolling Mean}_{w=3} = \left[ \frac{4.6+4.9+5}{3}, \frac{4.9+5+4.7}{3}, \frac{5+4.7+4.9}{3}, \frac{4.7+4.9+5.1}{3} \right] \]

\[ \text{Rolling Mean}_{w=3} = [4.83, 4.87, 4.87, 4.90] \]

Vamos considerar os valores reais de S1 a partir do quarto valor, pois a média móvel com w=3 só pode ser calculada a partir do terceiro valor.

\[ S1_{\text{real}} = [4.7, 4.9, 5.1] \]

### Cálculo do MAE para o modelo rolling mean com w=3
\[ \text{MAE}_{\text{Rolling Mean}} = \frac{1}{n} \sum_{i=1}^{n} |S1_{\text{real}_i} - \text{Rolling Mean}_{w=3_i}| \]

Cálculo dos MAEs

```python
import numpy as np

# Valores reais e previsões do modelo M
S1 = np.array([4.6, 4.9, 5, 4.7, 4.9, 5.1])
S2 = np.array([4.3, 5.1, 5, 4.9, 5, 5])

# Cálculo do MAE para o modelo M
mae_m = np.mean(np.abs(S1 - S2))

# Cálculo da média móvel com w=3
rolling_mean_w3 = np.array([np.mean(S1[i-3:i]) for i in range(3, len(S1)+1)])

# Valores reais correspondentes
S1_real = S1[3:]

# Cálculo do MAE para o modelo rolling mean com w=3
mae_rolling_mean = np.mean(np.abs(S1_real - rolling_mean_w3))

mae_m, mae_rolling_mean
```
Resultados
```python
mae_m = 0.23333333333333334
mae_rolling_mean = 0.13333333333333333
```

c) De acordo com o coeficiente de determinação (R2), o modelo rolling mean com w=3 é melhor do que o modelo M.



d) A distância DTW entre S1 e S2 é aproximadamente 0.23 (valor arredondado às centésimas).



D. Prediction
Considere a árvore de regressão aprendida sobre segmentos de 3 instantes de tempo de uma série temporal. T3
corresponde a T-3, T2 a T-2 e T1 a T-1.
a) A previsão da árvore para o valor correspondente ao registo [T1=70, T2=50, T3=50] é aproximadamente 70.
b) De acordo com a árvore, mais de 90% das previsões apenas dependem do último valor observado.
c) Se 17.5 fosse o valor mínimo para T1 em todo o conjunto de treino, então a previsão do valor seguinte para
[T1=17.5, T2=50, T3=50], de acordo com o KNN com K = 443 seria 24, desde que apenas se considerasse
cada segmento descrito por T1.
d) Se pretendesse treinar um perceptrão multi-camada (MLP) sobre segmentos de três instantes de tempo de
uma série temporal, seriam usadas obrigatoriamente três camadas escondidas (hidden layers)




# Estudo Exame


### Slides Times series
https://drive.google.com/file/d/1hg2e2iy3uYYtoV4u0OGaOf8VrLHKNVIz/view


### youtube video
https://www.youtube.com/watch?v=7UPgcI0ebi4

### How to Detect Seasonality, Trend, and Cyclic Patterns in Time Series Analysis


#### Seasonality
Seasonality refers to periodic fluctuations in a time series that occur at regular intervals, such as daily, monthly, or yearly. To detect seasonality:
- **Visual Inspection**: Plot the time series and look for repeating patterns at regular intervals.
- **Autocorrelation Function (ACF)**: Use ACF plots to identify significant spikes at specific lags that correspond to the seasonal period.
- **Seasonal Decomposition**: Apply methods like STL (Seasonal and Trend decomposition using Loess) to separate the seasonal component from the trend and residuals.

#### Trend
Trend indicates the long-term movement or direction in the time series data. To detect trends:
- **Visual Inspection**: Plot the time series and observe the overall direction (upward, downward, or constant).
- **Moving Averages**: Apply moving averages to smooth out short-term fluctuations and highlight the long-term trend.
- **Regression Analysis**: Fit a regression line to the time series data to quantify the trend.

#### Cyclic Patterns
Cyclic patterns are long-term oscillations that are not of fixed period, unlike seasonality. To detect cyclic patterns:
- **Visual Inspection**: Plot the time series and look for long-term cycles that do not follow a fixed period.
- **Fourier Transform**: Use Fourier analysis to identify dominant frequencies in the time series that correspond to cyclic behavior.
- **Spectral Analysis**: Apply spectral density estimation to detect cycles with varying periods.

By using these methods, you can effectively identify and analyze seasonality, trend, and cyclic patterns in time series data.

#### Stationary

#### Stationary Time Series
A stationary time series is one whose statistical properties such as mean, variance, and autocorrelation are constant over time. In other words, the series does not exhibit trends, seasonal effects, or other structures that change over time.

To detect stationarity:
- **Visual Inspection**: Plot the time series and look for constant mean and variance over time.
- **Summary Statistics**: Calculate and compare the mean and variance over different time intervals.
- **Autocorrelation Function (ACF)**: Check if the ACF plot shows a rapid decay, indicating stationarity.
- **Statistical Tests**: Apply tests like the Augmented Dickey-Fuller (ADF) test or the Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test to statistically determine if a time series is stationary.

By ensuring a time series is stationary, you can apply various time series forecasting models more effectively, as many models assume stationarity in the data.



# Resources on Time Series

## Time Series Decomposition and patterns
- https://www.youtube.com/watch?v=7UPgcI0ebi4
- https://www.youtube.com/watch?v=_z-a6WoNC2s
- https://www.youtube.com/watch?v=ca0rDWo7IpI


# LSTM
- https://www.youtube.com/watch?v=94PlBzgeq90
- https://www.youtube.com/watch?v=b61DPVFX03I
- https://www.youtube.com/watch?v=S8tpSG6Q2H0