import numpy
import pandas

from matplotlib import pyplot
import seaborn

students = pandas.read_csv("conjuntos/student_habits_performance.csv")

print(students)

# Nivel educativo de los padres contra puntuación del examen
figure, axes = pyplot.subplots(1, 2, figsize=(20, 10))

seaborn.kdeplot(x=students["exam_score"], 
                hue=students["parental_education_level"],
                ax=axes[0], fill=True)

seaborn.boxplot(x=students["parental_education_level"],
                y=students["exam_score"],
                hue=students["parental_education_level"],
                ax=axes[1])

pyplot.savefig("shared/output/g1.png", dpi=300)