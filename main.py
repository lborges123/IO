import pulp
import header

#DADOS

custo_equipa = header.readArrayFromFile('eqvendas.txt') #custo equipa de vendas
horasdisp = header.readArrayFromFile('horasdisp.txt') #horas disponiveis
horas = header.readMatrixFromFile('horas.txt') #horas no setor s para produzir f
lucro = header.readArrayFromFile('lucro.txt') #lucro
procura = header.readArrayFromFile('procura.txt') #procura
minprod = header.readIntegerFromFile('minprod.txt') #minimo de producao

#CONJUNTOS

numFam = range(0,len(procura)) 
numSet = range(0,len(horasdisp)) 

#VARIAVEIS DE DECISAO
x = pulp.LpVariable.dicts('xf',numFam,cat=pulp.LpContinuous,lowBound=0) #Quantidade produzida da familia j (xi >= 0)
ev = pulp.LpVariable.dicts('ev',numFam,cat=pulp.LpBinary) #Se a familia tem equipa de vendas associada ou nao  (ev = 0 || ev = 1 )

#CRIAR MODELO
modelo = pulp.LpProblem('producao', pulp.LpMaximize)

#FUNCAO OBJETIVO
modelo += (sum(lucro[i]*x[i] for i in numFam) - sum(ev[i]*custo_equipa[i] for i in numFam))

#RESTRICOES

for j in numSet:
  modelo += sum(horas[j][i]*x[i] for i in numFam) <= horasdisp[j],'LimiteHoras_{}'.format(j)

for i in numFam:
  modelo += x[i] >= minprod*ev[i],'MinimoProducao_{}'.format(i)

for i in numFam:
  modelo += x[i] <= procura[i]*ev[i],'ProcuraMaxima_{}'.format(i)
   
#RESOLVER
status=modelo.solve() 

if status==pulp.LpStatusOptimal:
  print (modelo.objective.value()) #imprime valor FO otimo na janela de output
  for i in numFam:
    print(ev[i], ev[i].varValue,'/', x[i], x[i].varValue) #imprime nome e valor no otimo da variavel y
  modelo.writeLP('mylp') #cria ficheiro LP do modelo