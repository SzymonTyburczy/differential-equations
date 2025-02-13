# problem numer 1 - rownanie transportu ciepla
import numpy as np
import matplotlib.pyplot as plt

MIN_INTEGRATION_POINTS = 2500
LOWER_BOUND = 0.0
UPPER_BOUND = 2.0
WYNIK = []
n = int(input("Podaj n - ilosc elementow w MES: "))


# Funkcja k(x) dana w zadaniu
def k(x):
    if 1.0 >= x >= 0.0:
        return 1
    elif 2.0 >= x > 1.0:
        return 2.0


# Funkcja ei dana na cwiczeniach
def ei(xi, h, x):
    if xi - h <= x < xi:
        return (x - (xi - h)) / h
    elif xi <= x <= xi + h:
        return ((xi + h) - x) / h
    else:
        return 0.0


# Funkcja ei pochodna
def eprim(xi, h, x):
    if xi > x >= xi - h:
        return 1.0 / h
    elif xi + h >= x >= xi:
        return -1.0 / h
    else:
        return 0.0


# tu sie dzieje cala magia
def main(n):
    # Alokacja tablic: macierz A, wektor wyrazów wolnych f oraz wektor rozwiązań u
    A = np.zeros((n, n), dtype=float)  # macierz kwadratowa n na n wypelniona zerami
    f_col = np.zeros(n, dtype=float)  # macierz 1 na n funkcji wypelniona zerami
    u_col = np.zeros(n, dtype=float)  # macierz 1 na n u wypelniona zerami

    # Inicjalizacja wektora prawej strony za pomoca odpowiednich wartosci
    f_col[0] = -20

    # liczmy h ktore jest rowne 2/n
    h = (UPPER_BOUND - LOWER_BOUND) / n
    build_matrix(A, n, h)

    # Eliminacja Gaussa (postać schodkowa)
    Row_Echelon_Form(A, f_col)

    # Rozwiązanie układu
    solve(A, f_col, u_col)

    for i in range(n):
        WYNIK.append((i, u_col[i]))


def build_matrix(A, n, h):
    """
    Wypełnienie macierzy A odpowiednimi współczynnikami:
    - A[0,0] i A[0,1], A[n-1,n-2], A[n-1,n-1],
    - Wewnątrz pętli trojdiagonalnej: i-1, i, i+1.
    """
    A[0, 0] = calculate(0, 0, h, n)
    if n > 1:
        A[0, 1] = calculate(0, 1, h, n)

    for i in range(1, n - 1):
        for j in range(i - 1, i + 2):  # te 3 obok siebie
            A[i, j] = calculate(i, j, h, n)

    if n > 1:
        A[n - 1, n - 2] = calculate(n - 1, n - 2, h, n)
        A[n - 1, n - 1] = calculate(n - 1, n - 1, h, n)


def calculate(row, col, h, n):
    return -ei(col * h, h, 0.0) * ei(row * h, h, 0.0) + quad_trapezoid(k, eprim, row, col, h, n)


def quad_trapezoid(f1, f3, row, col, h, n):
    if max(row, col) != 0:
        lower = max(row, col) - 1
    else:
        lower = 0
    upper = min(row, col) + 1
    if lower >= upper:
        return 0.0
        
    a = lower * h
    b = upper * h
    
    integration_points = max(MIN_INTEGRATION_POINTS, n)

    ih = (b - a) / (integration_points+2)

    def integrand(x):
        return f1(x) * f3(col * h, h, x) * f3(row * h, h, x)

    x_first = a + 1 * ih
    x_last = a + (integration_points + 1) * ih

    sum_val = integrand(x_first) + integrand(x_last)

    for i in range(2, integration_points + 1):
        x_i = a + i * ih
        sum_val += 2.0 * integrand(x_i)

    return 0.5 * ih * sum_val


def Row_Echelon_Form(A, f_col):
    # Przekształca macierz do postaci schodkowej (eliminacja Gaussa w wersji 'prostej')
    n = A.shape[0]
    for i in range(n - 1):
        pivot = A[i, i]
        j = i + 1
        # obliczamy współczynnik proporcji do wyzerowania A[j,i]
        prop = A[j, i] / pivot if pivot != 0 else 0.0
        A[j, i] = 0.0
        # zerujemy resztę w wierszu j
        for k in range(j, n):
            A[j, k] -= prop * A[i, k]
        f_col[j] -= prop * f_col[i]


def solve(A, f_col, u_col):
    """
    Rozwiązuje układ równań A u = f (macierz już w postaci schodkowej).
    """
    n = A.shape[0]
    for i in range(n-1, -1, -1):
        ssum = f_col[i]
        # Odejmujemy znane już składniki z prawej strony
        for j in range(i + 1, n):
            ssum -= A[i, j] * u_col[j]
        # Wyliczamy u[i]
        u_col[i] = ssum / A[i, i]


def main2():
    u_values = []
    for i in range(len(WYNIK)):
        u_values.append(WYNIK[i][1])
    n = len(u_values)
    x_values = np.linspace(0.0, 2.0, n)
    plt.plot(x_values, u_values)
    plt.grid()
    plt.title('Równanie transportu ciepła, n = ' + str(n))
    plt.xlabel('iksy')
    plt.ylabel('u(x)')
    plt.show()


main(n)
main2()
