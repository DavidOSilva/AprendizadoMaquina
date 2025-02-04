from colorama import Fore, Style

# Definindo para cada atributo de qual tipo e escala, respectivamente, ele é com base na nossa discussão.
sol = {
    'age': ['Quantitativo', 'Racional'],
    'sex': ['Qualitativo', 'Nominal'],
    'cp': ['Qualitativo', 'Nominal'], # Tipo de dor no peito (1 = angina típica, 2 = angina atípica, 3 = dor não anginosa e 4 = assintomático)
    'trestbps': ['Quantitativo', 'Racional'], # Pressão arterial em mm Hg.
    'chol': ['Quantitativo', 'Intervalar'], # Colesterol em mg/dl.
    'fbs': ['Qualitativo', 'Nominal'], # Presença de açúcar no sangue em jejum.
    'restecg': ['Qualitativo', 'Nominal'], # Resultados do eletrocardiograma em repouso (0 = Normal, 1 = anormalidade na onda ST-T e 2 = hipertrofia ventricular esquerda provável ou definitiva)
    'thalach': ['Quantitativo', 'Racional'], # Frequência cardíaca máxima atingida.
    'exang': ['Qualitativo', 'Nominal'], # Angina, uma dor no peito temporária, induzida por exercício.
    'oldpeak': ['Quantitativo', ' Intervalar'], # Depressão do segmento ST, redução de oxigênio para o músculo cardíaco, induzida por exercício.
    'slope': ['Qualitativo', 'Ordinal'], # Inclinação do segmento ST de pico do exercício (1 = ascendente, 2 = plano e 3 = descendente)
    'ca': ['Quantitativo', 'Ordinal'], # Número de vasos principais coloridos por fluoroscopia.
    'thal': ['Qualitativo', 'Nominal'], # (3 = normal; 6 = defeito fixo; 7 = defeito reversível)
    'target': ['Quantitativo', 'Ordinal']  # Neste caso, estamos nos referindo ao atributo alvo ou rótulo. É um valor inteiro, de 0 (sem presença) a 4 para diagnóstico de doença cardíaca.
}

def printColorido(message, color, end='\n'):
    color_map = { # Dicionário para mapear as cores fornecidas ao colorama
        'red': Fore.RED,
        'green': Fore.GREEN,
        'yellow': Fore.YELLOW,
        'blue': Fore.BLUE,
        'magenta': Fore.MAGENTA,
        'cyan': Fore.CYAN,
        'white': Fore.WHITE
    }
    if color in color_map: color_code = color_map[color] # Verifica se a cor fornecida está no dicionário.
    else:
        print("Cor inválida. Usando a cor padrão.")
        color_code = Fore.RESET
    print(f"{color_code}{message}{Style.RESET_ALL}", end=end) # Imprime a mensagem com a cor desejada.
    

def obterTodosValoresPossiveis(x, y):
    valores = {}
    for index, linha in x.iterrows():
        linha = linha.to_dict()  # Converte a linha em um dicionário
        for a, valor in linha.items():
            if a not in valores: valores[a] = set()
            valores[a].add(valor)
    for index, linha in y.iterrows():
        linha = linha.to_dict()  # Converte a linha em um dicionário
        for a, valor in linha.items():
            if a not in valores: valores[a] = set()
            valores[a].add(valor)
    for atributo in valores.keys(): valores[atributo] = sorted(valores[atributo])
    return valores

def contarClasses(y):
    classes = {}
    for index, linha in y.iterrows():
        linha = linha.to_dict() # Teremos algo assim tem cada linha: {'num': 0}
        classe = list(linha.values())[0] # Desestrutura diretamente o valor único da linha.
        if classe not in classes: classes[classe] = 1
        else: classes[classe] += 1
    return classes

def obterFrequenciaPorAtributo(array):
    frequencias = {}
    for valor in array:
        if valor not in frequencias: frequencias[valor] = 1
        else: frequencias[valor] += 1
    return frequencias

def obterIncompletude(df):
    linhasIncompletas = []
    incompletos = df.isnull().sum()
    for atributo, faltantes in incompletos.items(): # Obtém os índices das linhas com valores ausentes.
        if faltantes > 0: linhasIncompletas.extend(df.index[df[atributo].isnull()].tolist())
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(linhasIncompletas))
    printColorido(f"INDICES: ", "yellow", "")
    print(linhasIncompletas)
    
def obterInconsistencias(x):
    inconsistentes = []
    for (xi, xl) in x.iterrows():
        if xl['exang'] == 0 and xl['cp'] == 1: inconsistentes.append(xi) # Sem angina, mas com dor angina tipica.
        if xl['exang'] == 1 and xl['cp'] == 1 and xl['restecg'] == 0: inconsistentes.append(xi)
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(inconsistentes))
    printColorido(f"INDICES: ", "yellow", "")
    print(inconsistentes)
    
def obterRedundancias(x):
    duplicadas = x.index[x.duplicated(keep=False)].tolist()
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(duplicadas))
    printColorido(f"INDICES: ", "yellow", "")
    print(duplicadas)
    
def obterRuidos(x):
    ruidos = []
    for xi, xl in x.iterrows():
        if xl['thalach'] >= 200 or xl['thalach'] <= 0: ruidos.append(xi)
        if xl['chol'] >= 450 or xl['chol'] <= 0: ruidos.append(xi)
        if xl['trestbps'] >= 250 or xl['trestbps'] <= 0: ruidos.append(xi)
        if xl['oldpeak'] >= 6 or xl['oldpeak'] < 0: ruidos.append(xi)
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(ruidos))
    printColorido(f"INDICES: ", "yellow", "")
    print(ruidos)

