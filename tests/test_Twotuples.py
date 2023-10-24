import pytest
from Twotuples.Twotuples import PysentimentClasificator

@pytest.mark.parametrize('data,ColumnName, expected', [
   ('pruebadataset.xlsx','Opinion', 'Success'), #dataset correcto
   ('pruebadataset.xlsx','Class','Failed'),   #columna de clasificacion incorrecta
   ('prueba.xlsx','Opinion','Failed'),        #dataset que no existe
   ('pruebadataset_2".xlsx','Opiniones','Success'), #otro dataset correcto
])

def test_PysentimentClasificator(data,ColumnName,expected):
   assert PysentimentClasificator(data,ColumnName)==expected
    
    
    