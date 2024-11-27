# prediccion_salarios

## Contenidos
- [Alcance](#Alcance)
- [Datos](#Datos)

## Alcance
El proyecto comprende la implementación de modelos de machine learning supervisados y su posterior evaluación y selección para incorporar en una aplicación. La aplicación se trata de un tablero de control a través del cual tanto usuarios individuales como empresas puedan predecir e identificar tendencias salariales en los roles objeto de este proyecto.  
Para la implementación de los modelos y de la aplicación está se incorporarán prácticas de MLops y  de despliegue de aplicaciones cómo son: versionamiento de datos y del código, integración mediante APIs y plataformas abordadas en el curso que se irán incorporando a medida que avance el ciclo y logremos un entendimiento que nos permita elegir las herramientas a utilizar. 

##Datos
Los datos consisten en información salarial de roles relacionados con AI y Machine Learning, extraida de la pagina https://aijobs.net/salaries/download/, estos son de dominio público, y estan licenciados por CC0 1.0 (Universal Deed) lo cual significa que aijobs.net, la compañía que los administra, ha renunciado a sus derechos de autor, de modo que éstos pueden ser copiados, modificados, distribuidos y utilizados aún con fines comerciales sin necesidad de pedir permiso. Asimismo, estos datos no tienen garantía alguna, y aijobs.net no tiene ninguna responsabilidad o aprobación sobre el uso que den terceros a éstos. A continuación se presentan las variables que conforman los datos:
Año en el cual se recibe el salario, Nivel de experiencia (Junior, Intermedio, Experto,  Director/Ejecutivo), Tipo de Empleo (Medio tiempo, Tiempo completo, Contrato, Freelance), Título de la Posición (varias opciones), Salario (moneda local), Moneda salario, Salario en USD (calculado por aijobs.net), Residencia del empleado, Tasa de trabajo remoto (Presencial: Menos 20%, Hibrido: Entre el 20% y 80%, Remoto: Más del 80%), Ubicación de la Compañía (Oficinas Principal, o sucursal que contrata), Tamaño de la Compañía (Pequeña: Menos de 50 empleados, Mediana: 50 - 250 empleados, Grande: Más de 250 empleados)
