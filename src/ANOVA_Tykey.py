from scipy import stats
from statsmodels.stats.multicomp import MultiComparison
import codecs
import simplejson as json

print('Accuracy')

path_to_data = 'D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/statistical_tests_data.json'
with codecs.open(path_to_data, 'r') as f:
    data = f.read()
    data = json.loads(data)

for k in data:
    values={}
    mesure = data[k]
    values[k]={}

    values[k]['SGD'] = [m['value'] for m in mesure if m['key'] == 'SGD']
    values[k]['LSV'] = [m['value'] for m in mesure if m['key'] == 'LSV']
    values[k]['PA'] = [m['value'] for m in mesure if m['key'] == 'PA']
    values[k]['COMP'] = [m['value'] for m in mesure if m['key'] == 'COMP']



    f, p = stats.f_oneway( values[k]['SGD'],
                           values[k]['LSV'],
                           values[k]['PA'] ,
                           values[k]['COMP']
                           )

    print ('One-way ANOVA')
    print ('=============')
    print ('F value:', f)
    print ('P value:', p, '\n')

    mc = MultiComparison([int(m['value']) for m in mesure], [m['key'] for m in mesure])
    result = mc.tukeyhsd()
    print(result)
    print(mc.groupsunique)
