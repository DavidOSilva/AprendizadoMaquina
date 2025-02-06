from colorama import Fore, Style

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
    return linhasIncompletas
    
def obterRedundancias(x):
    duplicadas = x.index[x.duplicated(keep="first")].tolist()
    printColorido(f"QUANTIDADE: ", "yellow", "")
    print(len(duplicadas))
    printColorido(f"INDICES: ", "yellow", "")
    print(duplicadas)
    return duplicadas

