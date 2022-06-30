#!/usr/bin/env python
# coding: utf-8

# ## AUMENTO DO PREÇO DOS COMBUSTÍVEIS AO LONGO DOS ANOS NO BRASIL 

# ### Pós-graduação Lato Sensu em Ciência de Dados e Big Data
# #### Aluna: Jessica Rosa da Silva Salvador

# In[108]:


import numpy as np
import pandas as pd
import requests

get_ipython().system('pip install plotly')
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SKLearning - Utilizada para treinar e testar a regressão linear executada no projeto
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')


# In[109]:


def defDatasetTamanho(df):
    shape = df.shape
    print("O dataset apresenta " + str(shape[1]) + " colunas e " + str(shape[0]) + " registros.\n")
    
def deftiposeNulos(df):
    print("A seguir a listagem da tipagem das colunas e seu nível de preenchimento:\n")
    
    #display(df.columns)
    # info on variable types and filling factor
    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values'}))
    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100).T.
                         rename(index={0:'null values (%)'}))
    display(tab_info.transpose())
    


# ### Importando a base de dados dos preços de combustíveis(período de 2011 até 2022)

# In[110]:


base1 = pd.read_excel('mensal-estados-2001-a-2012.xlsx')


# In[111]:


base2 = pd.read_excel('mensal-estados-desde-jan2013.xlsx')


# ### Concatenação e Tratamento da base de dados de combustíveis

# In[112]:


#União dos dois dataframes utilizando do concat
base_df = pd.concat([base1, base2], axis=0)


# In[113]:


defDatasetTamanho(base_df)
deftiposeNulos(base_df)


# #### Correções de tipagem e valores errados

# In[114]:


# Acertando a tipagem das colunas para poder unir os dois dataframes 
# Como tem valores com '-' os mesmos ficam nulos quando acontece a conversão
base_df['PREÇO MÉDIO DISTRIBUIÇÃO'] = pd.to_numeric(base_df['PREÇO MÉDIO DISTRIBUIÇÃO'], errors='coerce')


# In[115]:


#Temos produtos repetidos por falta de acento nas palavras
base_df["PRODUTO"].unique()


# In[116]:


#Acertando os valores
base_df = base_df.replace("OLEO DIESEL", "ÓLEO DIESEL")
base_df = base_df.replace("OLEO DIESEL S10", "ÓLEO DIESEL S10")
base_df = base_df.replace("AMAPA", "AMAPÁ") 
base_df = base_df.replace("CEARA", "CEARÁ")
base_df = base_df.replace("ESPIRITO SANTO", "ESPÍRITO SANTO")
base_df = base_df.replace("GOIAS", "GOIÁS")
base_df = base_df.replace("MARANHAO", "MARANHÃO")
base_df = base_df.replace("PARA", "PARÁ")
base_df = base_df.replace("PARAIBA", "PARAÍBA")
base_df = base_df.replace("PARANA", "PARANÁ")
base_df = base_df.replace("PIAUI", "PIAUÍ")
base_df = base_df.replace("RONDONIA", "RONDÔNIA")
base_df = base_df.replace("SAO PAULO", "SÃO PAULO")


# In[117]:


#Conferindo os produtos após o tratamento
base_df["PRODUTO"].unique()


# In[118]:


#Criando a coluna ano
base_df['ANO'] = base_df['MÊS'].dt.year


# In[119]:


# Cria a coluna DATA removendo os dias
base_df['MÊS'] = base_df['MÊS'].dt.to_period('M')


# In[120]:


#excluindo colunas que não serão usadas do dataframe
base_df = base_df.drop(["NÚMERO DE POSTOS PESQUISADOS","UNIDADE DE MEDIDA","DESVIO PADRÃO REVENDA","PREÇO MÍNIMO REVENDA", 
                        "PREÇO MÁXIMO REVENDA","MARGEM MÉDIA REVENDA","DESVIO PADRÃO DISTRIBUIÇÃO","PREÇO MÍNIMO DISTRIBUIÇÃO",
                        "PREÇO MÁXIMO DISTRIBUIÇÃO","COEF DE VARIAÇÃO DISTRIBUIÇÃO","COEF DE VARIAÇÃO REVENDA"], axis=1)


# #### Filtragem

# In[121]:


#removendo 2011 para trás
dfremove = base_df.loc[(base_df['MÊS']<= '2007-12')]
base_df = base_df.drop(dfremove.index)

#Removendo o combustível GLP
dfremove_glp = base_df.loc[(base_df['PRODUTO'] == 'GLP')]
base_df = base_df.drop(dfremove_glp.index)
base_df['PRODUTO'].unique()


# #### Tratamento de dados ausentes

# In[122]:


deftiposeNulos(base_df)


# In[123]:


# Colocando o valor da mediana no lugar dos valores nulos
base_df['PREÇO MÉDIO DISTRIBUIÇÃO'].fillna(base_df["PREÇO MÉDIO DISTRIBUIÇÃO"].median(), inplace = True)


# In[124]:


#Verificação de dados duplicados, resultado: 0
linhas_dup_combs = base_df[base_df.duplicated(keep=False)]
linhas_dup_combs.shape


# In[125]:


base_df.tail(20)


# In[126]:


#Verificando 27 estados
base_df['ESTADO'].unique()


# # Dataset Municípios

# ### Importando o dataset da relação de municípios

# In[127]:


# https://servicodados.ibge.gov.br/api/docs/localidades
ids_municipios = requests.get('https://servicodados.ibge.gov.br/api/v1/localidades/distritos').json()


# In[128]:


#Importando todos os estados e UF
siglas_estados = []
nome_estados = []

for state in ids_municipios:
    siglas_estados.append(state['municipio']['microrregiao']['mesorregiao']['UF']['sigla'])
    nome_estados.append(state['municipio']['microrregiao']['mesorregiao']['UF']['nome'])

municipios = pd.DataFrame( {'SIGLA_UF':siglas_estados, 'ESTADO':nome_estados } )
defDatasetTamanho(municipios)


# In[129]:


deftiposeNulos(municipios)


# # Tratamento da base de Municípios

# In[130]:


#Colocando todos os estados em maiusculos para o merge com a base principal
municipios['ESTADO'] = municipios['ESTADO'].str.upper()


# #### Tratamento de dados duplicados

# In[131]:


#Exemplo de município duplicado
municipios.loc[(municipios['ESTADO'])=='GOIÁS']


# In[132]:


#Verificação de dados duplicados: (ANTES): resultado: 7284
linhas_dup_municipio = municipios[municipios.duplicated(keep=False)]
linhas_dup_municipio.shape


# In[133]:


#Removendo linhas duplicadas do dataset de municipios
municipios_df = municipios.drop_duplicates()

#Após a remoção das duplicatas, resultado: 0
linhas_dup_municipio = municipios_df[municipios_df.duplicated(keep=False)]
linhas_dup_municipio.shape


# In[134]:


#Exemplo inicial mostrando apenas 1 único registro
municipios_df.loc[(municipios['ESTADO'])=='GOIÁS']


# In[135]:


display(municipios_df)
deftiposeNulos(municipios_df)


# #### Junção do dataframe com a base principal

# In[136]:


# merge dos dois dataframes em um único
base_comb_df = pd.merge(base_df, municipios_df, how='left', on=['ESTADO'])
display(base_comb_df)


# In[137]:


#Verificando 27 estados
base_comb_df['SIGLA_UF'].unique()


# # Dataset do preço do Barril

# ### Importando o dataset dos preços do Barril

# In[138]:


#base_prc_barril = pd.read_csv("ipeadata_preco_barril.csv")


# In[139]:


base_prc_barril = pd.read_csv("https://raw.githubusercontent.com/jrosasalvador/TCC_PUC_Minas_2022/main/Bases/ipeadata_preco_barril.csv")


# In[140]:


base_prc_barril.sample(n=5)


# In[141]:


defDatasetTamanho(base_prc_barril)
deftiposeNulos(base_prc_barril)


# # Tratamento base do prc do barril

# In[142]:


#Renomeando a coluna
base_prc_barril.rename(columns={'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366': 'PRC_BARRIL_BRUTO'}, inplace = True)

#Deletando colunas desnecessárias
prc_barril_df = base_prc_barril.drop(["Unnamed: 2"], axis=1)


# In[143]:


# Cria a coluna DATA removendo os dias
prc_barril_df['Data'] = pd.to_datetime(prc_barril_df['Data'])
prc_barril_df['Data'] = prc_barril_df['Data'].dt.to_period('M')


# #### Filtragem

# In[144]:


# Agrupa também as datas pelo mês, fazendo a média do valor
prc_barril_mes_df = prc_barril_df.groupby(['Data']).agg({'PRC_BARRIL_BRUTO': np.mean}).reset_index()


# In[145]:


# Define data como index da coluna para o merge posterior
prc_barril_mes_df = prc_barril_mes_df.set_index('Data')


# In[146]:


#Verificando existencia de duplicatas: resultado: 0
linhas_dup_barril = prc_barril_mes_df[prc_barril_mes_df.duplicated(keep=False)]
linhas_dup_barril.shape


# In[147]:


prc_barril_mes_df.head(5)


# # Dataset da Taxa de câmbio - Dólar americano

# ### Importando dataset da taxa de câmbio

# In[148]:


#df_cambio = pd.read_csv("Cotação do Dólar por período.csv")


# In[149]:


df_cambio = pd.read_csv("https://raw.githubusercontent.com/jrosasalvador/TCC_PUC_Minas_2022/main/Bases/Cotacao%20do%20Dolar%20por%20periodo.csv")


# In[150]:


defDatasetTamanho(df_cambio)
deftiposeNulos(df_cambio)


# In[151]:


df_cambio.head(5)


# # Tratamento da base da taxa de câmbio

# In[152]:


# Cria a coluna DATA removendo os dias
df_cambio['dataHoraCotacao'] = pd.to_datetime(df_cambio['dataHoraCotacao'])
df_cambio['dataHoraCotacao'] = df_cambio['dataHoraCotacao'].dt.to_period('M')


# In[153]:


#Deletando a coluna 'cotacaoCompra' pois não a usaremos para a análise
df_cambio = df_cambio.drop(["cotacaoCompra"], axis=1)


# #### Correções de tipagem

# In[154]:


#Como a coluna cotacaoVenda é do tipo object, preciso primeiramente substituir a vírgula pelo ponto
#Pois assim não dará erro na hora de converter seu valor para numérico
df_cambio['cotacaoVenda'] = df_cambio['cotacaoVenda'].apply(lambda x: float(x.replace(",",".")))


# In[155]:


#convertendo a coluna para numeric
df_cambio['cotacaoVenda'] = pd.to_numeric(df_cambio['cotacaoVenda'], errors='coerce')


# #### Filtragem

# In[156]:


# Agrupa por Ano/Mês, fazendo a média do valor da cotação da venda
df_tx_cambio = df_cambio.groupby(['dataHoraCotacao']).agg({'cotacaoVenda': np.mean}).reset_index()


# In[157]:


#Setando a colunadaHoraCotacao como índice para unir com a base principal posteriormente
df_tx_cambio = df_tx_cambio.set_index('dataHoraCotacao')
display(df_tx_cambio)


# In[158]:


#Verificação de dados duplicados. Resultado: 0
linhas_dup_cambio = df_tx_cambio[df_tx_cambio.duplicated(keep=False)]
linhas_dup_cambio.shape


# ## Agregando as colunas no dataframe principal

# #### Junção das principais colunas com o base principal

# In[159]:


# Copia os valores do dataframe para um dataframe definitivo
df_prc_comb = base_comb_df.copy()
# Reseta o index para um valor sequencial e depois seta a coluna DATA como novo index
df_prc_comb.reset_index(inplace=True)
df_prc_comb.set_index('MÊS', inplace=True)


# In[160]:


# Agrega as colunas dos datasets anteriores (Barril, Câmbio, Inflação...)


# In[161]:


df_prc_comb['PRC_BARRIL_BRUTO'] = prc_barril_mes_df['PRC_BARRIL_BRUTO']  


# In[162]:


df_prc_comb['TAXA CÂMBIO'] = df_tx_cambio['cotacaoVenda']


# In[163]:


# Deletando coluna desnecessária
df_prc_comb = df_prc_comb.drop(["index"], axis=1)


# In[164]:


df_prc_comb.tail(5)


# ## ANÁLISE DOS PREÇOS DOS COMBUSTÍVEIS

# In[165]:


# tabela estatística
estatisticas_df = df_prc_comb.groupby(['PRODUTO'], sort=False).agg({'PREÇO MÉDIO REVENDA': [np.min,np.max,np.mean,np.std]}).reset_index()
display(estatisticas_df)


# In[166]:


# Preço máximo de revenda por produto
vendas_produto_max_df = df_prc_comb.groupby(['PRODUTO'], sort=False).agg({'PREÇO MÉDIO REVENDA': np.max}).reset_index()
display(vendas_produto_max_df)


# In[167]:


# Gráfico do Preço máximo de revenda por produto
grafico_max = px.bar(vendas_produto_max_df, x="PREÇO MÉDIO REVENDA", y="PRODUTO", color = "PRODUTO", 
                   title = "Preço Máximo por Produto", height=400,text_auto=True,
                   labels={'PRODUTO':'Produto', 'PREÇO MÉDIO REVENDA':'Preço Máximo'})
grafico_max.show()


# In[168]:


# Preço médio de revenda por produto
vendas_produto_df = df_prc_comb[["PRODUTO","PREÇO MÉDIO REVENDA"]].groupby(["PRODUTO"]).mean().reset_index()
display(vendas_produto_df)


# In[169]:


grafico_mean = px.bar(vendas_produto_df, x="PRODUTO", y="PREÇO MÉDIO REVENDA", color = "PRODUTO", 
                   title = "Preço Médio de Revenda por Produto", text_auto='.3s',
                   labels={'PRODUTO':'Produto', 'PREÇO MÉDIO REVENDA':'Preço Médio Revenda'})
grafico_mean.update_traces(textangle=0, textposition="outside")
grafico_mean.show()


# In[170]:


#preço médio de revenda por produto e ano
vendas_produto_ano = df_prc_comb.groupby(['PRODUTO','ANO'], sort=False).agg({'PREÇO MÉDIO REVENDA': np.mean}).reset_index()
vendas_produto_ano.head()


# In[171]:


grafico_prod_ano = px.line(vendas_produto_ano, x="ANO", y="PREÇO MÉDIO REVENDA", color="PRODUTO",
                          title = "Preço Médio de Revenda por Ano",markers=True,
                   labels={'ANO':'Ano', 'PREÇO MÉDIO REVENDA':'Preço Médio Revenda'})
grafico_prod_ano.update_traces(textposition="bottom right")
grafico_prod_ano.show()


# In[172]:


#preço médio de revenda por região
vendas_regiao = df_prc_comb.groupby(["REGIÃO"], sort=False).agg({'PREÇO MÉDIO REVENDA': np.mean}).reset_index()
display(vendas_regiao)


# In[173]:


# Construção do gráfico para mostrar o preço médio de revenda por região
grafico_regiao = px.bar(vendas_regiao, x="REGIÃO", y="PREÇO MÉDIO REVENDA", color="REGIÃO", 
                                          title = "Preço Médio de Revenda por Região", text_auto='.3s',
                   labels={'REGIÃO':'Região', 'PREÇO MÉDIO REVENDA':'Preço Médio Revenda'})
grafico_regiao.update_traces(textangle=0, textposition="outside")
grafico_regiao.show()


# In[174]:


#preço médio de revenda por ano e regiao
vendas_ano_regiao = df_prc_comb.groupby(['REGIÃO','ANO'], sort=False).agg({'PREÇO MÉDIO REVENDA': np.mean}).reset_index()
vendas_ano_regiao.head()


# In[175]:


grafico_ano_regiao = px.bar(vendas_ano_regiao, y="ANO", x="PREÇO MÉDIO REVENDA", color="REGIÃO", orientation ='h',
                           title = "Preço Médio de Revenda por Ano e Região", text_auto='.3s',
                   labels={'REGIÃO':'Região', 'PREÇO MÉDIO REVENDA':'Preço Médio Revenda'})
grafico_ano_regiao.update_traces(textangle=0, textposition="outside")
grafico_ano_regiao.show()


# In[176]:


#preço médio de revenda por produto e regiao
vendas_prod_regiao = df_prc_comb.groupby(['REGIÃO','PRODUTO'], sort=False).agg({'PREÇO MÉDIO REVENDA': np.mean}).reset_index()
vendas_prod_regiao.head()


# In[177]:


grafico_prod_regiao = px.bar(vendas_prod_regiao, x="REGIÃO", y="PREÇO MÉDIO REVENDA", color="PRODUTO", barmode="group",
                             title = "Preço Médio de Revenda por Região e Produto", text_auto='.3s',
                   labels={'REGIÃO':'Região', 'PREÇO MÉDIO REVENDA':'Preço Médio Revenda'})
grafico_prod_regiao.update_traces(textangle=0, textposition="outside")
grafico_prod_regiao.show()


# In[178]:


#preço médio de revenda por estado e regiao
vendas_uf = df_prc_comb.groupby(["SIGLA_UF","PRODUTO"], sort=False).agg({'PREÇO MÉDIO REVENDA': np.mean}).reset_index()
vendas_uf.head()


# In[179]:


# Construção do gráfico para mostrar o preço médio de revenda estado e região
grafico_uf = px.bar(vendas_uf, x="SIGLA_UF", y="PREÇO MÉDIO REVENDA", color='PRODUTO',
                              title = "Preço Médio de Revenda por Região", text_auto='.3s',
                   labels={'SIGLA_UF':'UF', 'PREÇO MÉDIO REVENDA':'Preço Médio Revenda'})
grafico_uf.update_traces(textangle=0, textposition="outside")
grafico_uf.show()


# In[180]:


df_prc_comb.sample(5)


# In[181]:


#Agrupou o dataframe pela data e realizou a média de todos os outros campos
preco_anual_df = df_prc_comb.groupby(['ANO']).mean()


# In[182]:


preco_anual_df.sample(5)


# In[183]:


# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=preco_anual_df.index, y=preco_anual_df['PREÇO MÉDIO REVENDA'], name="preço revenda"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=preco_anual_df.index, y=preco_anual_df['PRC_BARRIL_BRUTO'], name="preço barril"), 
    secondary_y=True,
)



# Add figure title
fig.update_layout(
    title_text="Preço Médio de Revenda x Preço do Barril de Petróleo"
)

#fig.update_traces(textposition="middle center")

# Set x-axis title
fig.update_xaxes(title_text="Ano")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Preço Médio de Revenda</b>", secondary_y=False)
fig.update_yaxes(title_text="<b> Preço do Barril de Petróleo</b>", secondary_y=True)

fig.show()


# In[184]:


# Create figure with secondary y-axis
fig_2 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig_2.add_trace(
    go.Scatter(x=preco_anual_df.index, y=preco_anual_df['PREÇO MÉDIO REVENDA'], name="preço revenda"),
    secondary_y=False,
)

fig_2.add_trace(
    go.Scatter(x=preco_anual_df.index, y=preco_anual_df['TAXA CÂMBIO'], name="Taxa do Câmbio"),
    secondary_y=True,
)

# Add figure title
fig_2.update_layout(
    title_text="Preço Médio de Revenda x Taxa do Câmbio"
)

#fg.update_traces(trendline="ols")

# Set x-axis title
#fig.update_xaxes(title_text="Ano")

# Set y-axes titles
fig_2.update_yaxes(title_text="<b>Preço Médio de Revenda</b>", secondary_y=False)
fig_2.update_yaxes(title_text="<b>Taxa do Câmbio</b>", secondary_y=True)

fig_2.show()


# In[185]:


preco_anual_df.corr()


# In[186]:


plt.figure(figsize=(13,8))
sns.heatmap(preco_anual_df.corr(),annot=True)
plt.show()


# ### Machine learning

# In[187]:


df_prc_comb.head(5)


# In[188]:


models = {}


# In[189]:


reg_df = df_prc_comb[df_prc_comb.SIGLA_UF == 'RJ'].groupby(['PRODUTO', 'MÊS']).mean()


# In[190]:


reg_df = reg_df.iloc[reg_df.index.get_level_values('PRODUTO') == 'GASOLINA COMUM'].groupby('MÊS').mean()


# In[191]:


# Separa eixos da regressão
X = reg_df.drop(['PREÇO MÉDIO REVENDA','PREÇO MÉDIO DISTRIBUIÇÃO','ANO','PRC_BARRIL_BRUTO'] , axis =1)
Y = reg_df['PREÇO MÉDIO REVENDA'].values


# In[192]:


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.30)


# In[193]:


# Cria model de regressão e faz o treinamento
lr=LinearRegression()
lr.fit(x_train,y_train)

models = lr
predicted_values = []
for i in range(0, len(y_test)):
    predicted_values.append(lr.predict(x_test.iloc[[i],:])[0])


# In[194]:


predicted_df = pd.DataFrame({'Cambio':x_test['TAXA CÂMBIO'].values, 
                             'Valor Real':y_test, 'Valor Predito':predicted_values})


# In[195]:


predicted_df.head(5)


# In[196]:


r2_score(predicted_df["Valor Real"], predicted_df["Valor Predito"])


# In[197]:


predicted_df.sort_values(by=['Cambio']).set_index('Cambio')


# In[198]:


# Create figure with secondary y-axis
fig_3 = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig_3.add_trace(
    go.Bar(x=predicted_df.index, y=predicted_df['Valor Real'], name="Valor Real"),
    secondary_y=True,
)

fig_3.add_trace(
    go.Bar(x=predicted_df.index, y=predicted_df['Valor Predito'], name="Valor Predito"),
    secondary_y=False,
)

# Add figure title
fig_3.update_layout(
    title_text="Regressão - Gasolina Comum"
)

# Set x-axis title
fig_3.update_xaxes(title_text="Câmbio")

# Set y-axes titles
fig_3.update_yaxes(title_text="<b>Valor Real</b>", secondary_y=False)
fig_3.update_yaxes(title_text="<b>Valor Predito</b>", secondary_y=True)

fig_3.show()


# ### Randon forest regressor

# In[199]:


models_RF = {}


# In[200]:


# Separa eixos da regressão
X = reg_df.drop(['PREÇO MÉDIO REVENDA','PREÇO MÉDIO DISTRIBUIÇÃO','ANO','PRC_BARRIL_BRUTO'] , axis =1)
Y = reg_df['PREÇO MÉDIO REVENDA'].values


# In[201]:


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.30)


# In[202]:


rf = RandomForestRegressor()


# In[203]:


rf.fit(x_train, y_train)


# In[204]:


models_RF = rf
predicted_values_rf = []
for i in range(0, len(y_test)):
    predicted_values_rf.append(rf.predict(x_test.iloc[[i],:])[0])


# In[205]:


predicted_rf_df = pd.DataFrame({'Cambio_rf':x_test['TAXA CÂMBIO'].values, 
                            'Valor_Real':y_test, 'Valor_Predito':predicted_values_rf})


# In[206]:


predicted_rf_df.sample(5)


# In[207]:


r2_score(predicted_rf_df["Valor_Real"], predicted_rf_df["Valor_Predito"])


# ### Previsão dos valores

# In[208]:


resultado = {}


# In[209]:


resultado['2020-01'] = df_tx_cambio['cotacaoVenda'].iloc[-1]


# In[210]:


for i in range(2021, 2031):
    ano_passado = str(i - 1)+'-01'
    ano_corrente = str(i)+'-01'
    resultado[ano_corrente]  = (resultado[ano_passado] * 0.04) + resultado[ano_passado]


# In[211]:


resultado_df = pd.DataFrame(list(resultado.items()), columns = ['MÊS', 'TAXA CÂMBIO'])


# In[212]:


resultado_df['MÊS'] = pd.to_datetime(resultado_df['MÊS'])
resultado_df.set_index('MÊS', inplace=True)


# In[213]:


resultado_df['PREVISÃO'] = models.predict(resultado_df)


# In[214]:


resultado_df


# In[ ]:




