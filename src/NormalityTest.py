import scipy.stats as stats
import codecs
import simplejson as json

path_to_data = 'D:/Sasha/subversion/trunk/AuthorshipAttributionRussianTexts/results/statistical_tests_data.json'
with codecs.open(path_to_data, 'r') as f:
    data = f.read()
    data = json.loads(data)

shapiro_results={}

for k in data:
    values={}
    mesure = data[k]
    values[k]={}
#    for key in mesure:
    values[k]['SGD'] = [m['value'] for m in mesure if m['key'] == 'SGD']
    values[k]['LSV'] = [m['value'] for m in mesure if m['key'] == 'LSV']
    values[k]['PA'] = [m['value'] for m in mesure if m['key'] == 'PA']
    values[k]['COMP'] = [m['value'] for m in mesure if m['key'] == 'COMP']

    shapiro_results[k]={}

    '''
        shapiro_results[k]['SGD'] = stats.shapiro(values[k]['SGD'])
        shapiro_results[k]['LSV'] = stats.shapiro(values[k]['LSV'])
        shapiro_results[k]['PA'] = stats.shapiro(values[k]['PA'])
        shapiro_results[k]['COMP'] = stats.shapiro(values[k]['COMP'])
    '''
    shapiro_results[k]['SGD'] = stats.normaltest(values[k]['SGD'])
    shapiro_results[k]['LSV'] = stats.normaltest(values[k]['LSV'])
    shapiro_results[k]['PA'] = stats.normaltest(values[k]['PA'])
    shapiro_results[k]['COMP'] = stats.normaltest(values[k]['COMP'])

    #shapiro_results[k] = stats.normaltest(values[k]['SGD']+  values[k]['PA']+values[k]['LSV']+values[k]['COMP'] )

    
