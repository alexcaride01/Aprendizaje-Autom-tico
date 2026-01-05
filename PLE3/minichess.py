from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import yaml

# Configuración del tablero 4x4
N_FILAS = 4
N_COLS = 4

# Representación de las piezas
# Jugador 1 (positivos): Rey=6, Torre=5, Peón=1
# Jugador 2 (negativos): Rey=-6, Torre=-5, Peón=-1
# Casilla vacía = 0

PEON = 1
TORRE = 5
REY = 6


class JugadorMiniChess(ABC):
    """
    Clase abstracta que representa un jugador de MiniChess.
    """
    
    def __init__(self, nombre: str) -> None:
        """
        Inicializa un jugador con un nombre.
        
        :param nombre: Nombre del jugador.
        """
        self.name = nombre
        self._estados_juego_actual = []  # Conjunto de estados visitados en la partida en curso
    
    @abstractmethod
    def decide_accion(
        self, 
        movimientos_validos: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
        tablero: np.ndarray, 
        token: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Dado un tablero y un token, decide la próxima acción a realizar.
        
        :param movimientos_validos: Lista de movimientos válidos (origen, destino).
        :param tablero: Estado actual del tablero.
        :param token: Token del jugador actual (1 o -1).
        :return: Tupla con (posición_origen, posición_destino).
        """
        pass
    
    def reset(self):
        """
        Resetea el jugador para una nueva partida.
        """
        # Limpiamos la lista de estados para comenzar una nueva partida desde cero
        self._estados_juego_actual = []
    
    def guarda_politica(self, path_artefacto: Path):
        """
        Guarda la política del jugador en un fichero YAML.
        
        :param path_artefacto: Ruta donde se guardará el fichero YAML.
        """
        # Persistimos el diccionario de experiencias aprendidas en formato YAML
        with open(path_artefacto, "w") as f:
            yaml.dump(self._experiencia_estado_valor, f, Dumper=yaml.Dumper)
    
    def carga_politica(self, path_artefacto: Path):
        """
        Carga la política de un fichero YAML.
        
        :param path_artefacto: Ruta del fichero YAML.
        """
        # Cargamos las experiencias previamente guardadas para continuar el aprendizaje
        with open(path_artefacto, "r") as f:
            self._experiencia_estado_valor = yaml.load(f, Loader=yaml.FullLoader)


class JugadorMiniChessMaq(JugadorMiniChess):
    """
    Jugador de MiniChess controlado por IA con aprendizaje por refuerzo.
    """
    
    def __init__(
        self,
        nombre: str,
        tasa_exploracion: float = 0.3,
        tasa_aprendizaje: float = 0.2,
        descuento_gamma: float = 0.9,
    ):
        """
        Inicializa un jugador máquina.
        
        :param nombre: Nombre del jugador.
        :param tasa_exploracion: Ratio de exploración (epsilon). Por defecto, 0.3.
        :param tasa_aprendizaje: Tasa de aprendizaje (alpha) para actualización de valores.
        :param descuento_gamma: Factor de descuento gamma para recompensas futuras.
        """
        super().__init__(nombre)
        # Configuramos los hiperparámetros del algoritmo de aprendizaje por refuerzo
        self._tasa_exploracion = tasa_exploracion
        self._tasa_aprendizaje = tasa_aprendizaje
        self._descuento_gamma = descuento_gamma
        # Inicializamos el diccionario donde almacenaremos los valores aprendidos de cada estado
        self._experiencia_estado_valor = {}  # Diccionario: estado_hash -> valor
    
    def decide_accion(
        self, 
        movimientos_validos: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
        tablero: np.ndarray, 
        token: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Decide el próximo movimiento usando epsilon-greedy.
        """
        # Exploración: movimiento aleatorio
        # Implementamos una estrategia epsilon-greedy: con probabilidad epsilon exploramos aleatoriamente
        if np.random.uniform(0, 1) <= self._tasa_exploracion:
            return movimientos_validos[np.random.choice(len(movimientos_validos))]
        
        # Explotación: seleccionar el mejor movimiento según la política aprendida
        # Si no exploramos, elegimos el movimiento que nos lleve al estado con mayor valor
        valor_max = -np.inf
        mejores_movimientos = []
        
        # Evaluamos cada movimiento posible simulando su resultado
        for movimiento in movimientos_validos:
            origen, destino = movimiento
            # Simular el movimiento
            # Creamos una copia del tablero para simular el movimiento sin afectar el estado real
            siguiente_tablero = tablero.copy()
            pieza = siguiente_tablero[origen]
            siguiente_tablero[destino] = pieza
            siguiente_tablero[origen] = 0
            
            # Serializar el estado y obtener su valor
            # Convertimos el estado del tablero a un hash para buscarlo en nuestra tabla de valores
            siguiente_tablero_hash = JuegoMiniChess._serializa_estado(
                siguiente_tablero, N_FILAS, N_COLS
            )
            # Obtenemos el valor del estado resultante (0 si nunca lo hemos visitado)
            valor = self._experiencia_estado_valor.get(siguiente_tablero_hash, 0)
            
            # Actualizamos la lista de mejores movimientos si encontramos un valor superior
            if valor > valor_max:
                valor_max = valor
                mejores_movimientos = [movimiento]
            elif valor == valor_max:
                mejores_movimientos.append(movimiento)
        
        # Selección aleatoria entre los mejores movimientos
        # Si hay varios movimientos con el mismo valor máximo, elegimos uno al azar
        return mejores_movimientos[np.random.choice(len(mejores_movimientos))]
    
    def guarda_estado(self, s: str):
        """
        Guarda el estado actual en la lista de estados visitados de la partida.
        
        :param s: Hash del estado actual del tablero.
        """
        # Añadimos el estado a la lista para poder retropropagar recompensas al final de la partida
        self._estados_juego_actual.append(s)
    
    def retropropaga_recompensa(self, recompensa_final: float) -> None:
        """
        Retropropaga la recompensa final de la partida a todos los estados visitados.
        Implementa actualización temporal-difference (TD).
        
        :param recompensa_final: Recompensa final de la partida.
        """
        # Comenzamos con la recompensa final y la propagamos hacia atrás por todos los estados
        recompensa_actual = recompensa_final
        
        # Recorremos los estados en orden inverso (desde el final hasta el inicio de la partida)
        for i in reversed(range(len(self._estados_juego_actual))):
            s = self._estados_juego_actual[i]
            # Obtenemos el valor actual del estado (0 si es la primera vez que lo visitamos)
            valor_s = self._experiencia_estado_valor.get(s, 0)
            
            # Actualización TD: V(s) = V(s) + alpha * (R - V(s))
            # Aplicamos la regla de actualización temporal-difference para ajustar el valor del estado
            nuevo_valor = valor_s + (self._tasa_aprendizaje * (recompensa_actual - valor_s))
            self._experiencia_estado_valor[s] = nuevo_valor
            
            # Actualizar recompensa para el siguiente estado (con descuento)
            # Aplicamos el factor de descuento gamma para dar menos importancia a recompensas futuras
            recompensa_actual = nuevo_valor * self._descuento_gamma


class JugadorMiniChessHum(JugadorMiniChess):
    """
    Jugador humano de MiniChess.
    """
    
    def decide_accion(
        self, 
        movimientos_validos: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
        tablero: np.ndarray, 
        token: int
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Solicita al jugador humano que introduzca un movimiento válido.
        """
        # Mostramos al usuario todos los movimientos posibles con un índice numérico
        print(f"\nMovimientos válidos disponibles:")
        for idx, (origen, destino) in enumerate(movimientos_validos):
            print(f"  {idx}: Desde {origen} hasta {destino}")
        
        # Pedimos al usuario que seleccione un movimiento hasta que introduzca uno válido
        while True:
            try:
                opcion = input("\nIntroduzca el número del movimiento que desea realizar: ").strip()
                idx = int(opcion)
                
                # Verificamos que el índice esté dentro del rango válido
                if 0 <= idx < len(movimientos_validos):
                    return movimientos_validos[idx]
                else:
                    raise ValueError("Índice fuera de rango.")
            except:
                # Si hay un error en la entrada, pedimos que lo intente de nuevo
                print("Entrada no válida. Inténtelo de nuevo...")
                continue


class JuegoMiniChess:
    """
    Clase que gestiona el juego de MiniChess 4x4 con aprendizaje por refuerzo.
    """
    
    def __init__(
        self, 
        jugador1: JugadorMiniChessMaq, 
        jugador2: JugadorMiniChess
    ) -> None:
        """
        Inicializa el juego con dos jugadores.
        
        :param jugador1: Jugador 1 (siempre máquina).
        :param jugador2: Jugador 2 (máquina o humano).
        """
        # Guardamos las referencias a los dos jugadores
        self._jugador1 = jugador1
        self._jugador2 = jugador2
        # Establecemos las dimensiones del tablero
        self._n_filas = N_FILAS
        self._n_cols = N_COLS
        # Inicializamos el tablero con las piezas en su posición inicial
        self._tablero = self.__inicializar_tablero()
        # Variables de control del estado del juego
        self._fin = False
        self._estado = None
        self._siguiente_jugador = 1  # Empieza el jugador 1
        # Contador para detectar empates por falta de progreso
        self._turnos_sin_captura = 0
    
    def __inicializar_tablero(self) -> np.ndarray:
        """
        Inicializa el tablero 4x4 con la configuración inicial de piezas.
        
        Configuración:
        Fila 0 (Jugador 1): Torre, Rey, vacío, vacío
        Fila 1 (Jugador 1): Peón, vacío, vacío, vacío
        Fila 2 (Jugador 2): vacío, vacío, vacío, Peón
        Fila 3 (Jugador 2): vacío, vacío, Rey, Torre
        """
        # Creamos un tablero vacío
        tablero = np.zeros((N_FILAS, N_COLS))
        
        # Jugador 1 (parte superior, valores positivos)
        # Colocamos las piezas del jugador 1 en la parte superior del tablero
        tablero[0, 0] = TORRE      # Torre en esquina superior izquierda
        tablero[0, 1] = REY        # Rey junto a la torre
        tablero[1, 0] = PEON       # Peón delante de la torre
        
        # Jugador 2 (parte inferior, valores negativos)
        # Colocamos las piezas del jugador 2 en la parte inferior con valores negativos (simetría diagonal)
        tablero[3, 3] = -TORRE     # Torre en esquina inferior derecha
        tablero[3, 2] = -REY       # Rey junto a la torre
        tablero[2, 3] = -PEON      # Peón delante de la torre
        
        return tablero
    
    @staticmethod
    def _serializa_estado(tablero: np.ndarray, n_filas: int, n_cols: int) -> str:
        """
        Serializa el tablero a una representación en string para almacenamiento.
        
        :param tablero: Tablero actual.
        :param n_filas: Número de filas.
        :param n_cols: Número de columnas.
        :return: String con la representación del tablero.
        """
        # Convertimos el tablero a un vector unidimensional y lo convertimos a string
        # Esto nos permite usarlo como clave en el diccionario de experiencias
        return str(tablero.reshape(n_cols * n_filas))
    
    def __calcular_movimientos_validos(
        self, 
        tablero: np.ndarray, 
        jugador: int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calcula todos los movimientos válidos para el jugador actual.
        
        :param tablero: Estado actual del tablero.
        :param jugador: 1 para jugador 1, -1 para jugador 2.
        :return: Lista de movimientos válidos (origen, destino).
        """
        # Lista donde acumularemos todos los movimientos posibles
        movimientos = []
        
        # Recorremos todo el tablero buscando piezas del jugador actual
        for i in range(N_FILAS):
            for j in range(N_COLS):
                pieza = tablero[i, j]
                
                # Solo procesar piezas del jugador actual
                # Verificamos que la pieza pertenezca al jugador actual (mismo signo)
                if (jugador == 1 and pieza > 0) or (jugador == -1 and pieza < 0):
                    # Obtenemos el tipo de pieza sin el signo
                    tipo_pieza = abs(pieza)
                    origen = (i, j)
                    
                    # Calculamos los movimientos según el tipo de pieza
                    if tipo_pieza == PEON:
                        movimientos.extend(self.__movimientos_peon(tablero, origen, jugador))
                    elif tipo_pieza == TORRE:
                        movimientos.extend(self.__movimientos_torre(tablero, origen, jugador))
                    elif tipo_pieza == REY:
                        movimientos.extend(self.__movimientos_rey(tablero, origen, jugador))
        
        return movimientos
    
    def __movimientos_peon(
        self, 
        tablero: np.ndarray, 
        origen: Tuple[int, int], 
        jugador: int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calcula movimientos válidos para un peón.
        """
        movimientos = []
        i, j = origen
        # El jugador 1 avanza hacia abajo (+1), el jugador 2 hacia arriba (-1)
        direccion = 1 if jugador == 1 else -1
        
        # Movimiento hacia adelante (una casilla)
        # Comprobamos si puede avanzar una casilla hacia adelante (casilla vacía)
        nueva_i = i + direccion
        if 0 <= nueva_i < N_FILAS and tablero[nueva_i, j] == 0:
            movimientos.append((origen, (nueva_i, j)))
        
        # Captura diagonal (izquierda)
        # Los peones capturan en diagonal, comprobamos la diagonal izquierda
        nueva_j = j - 1
        if (0 <= nueva_i < N_FILAS and 0 <= nueva_j < N_COLS and
            tablero[nueva_i, nueva_j] * jugador < 0):  # Pieza enemiga
            movimientos.append((origen, (nueva_i, nueva_j)))
        
        # Captura diagonal (derecha)
        # Comprobamos la diagonal derecha para posibles capturas
        nueva_j = j + 1
        if (0 <= nueva_i < N_FILAS and 0 <= nueva_j < N_COLS and
            tablero[nueva_i, nueva_j] * jugador < 0):  # Pieza enemiga
            movimientos.append((origen, (nueva_i, nueva_j)))
        
        return movimientos
    
    def __movimientos_torre(
        self, 
        tablero: np.ndarray, 
        origen: Tuple[int, int], 
        jugador: int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calcula movimientos válidos para una torre (horizontal y vertical).
        """
        movimientos = []
        i, j = origen
        
        # Direcciones: arriba, abajo, izquierda, derecha
        # Definimos las cuatro direcciones en las que se puede mover una torre
        direcciones = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Exploramos cada dirección hasta encontrar un obstáculo o el borde del tablero
        for di, dj in direcciones:
            nueva_i, nueva_j = i + di, j + dj
            
            # Avanzamos en esta dirección mientras estemos dentro del tablero
            while 0 <= nueva_i < N_FILAS and 0 <= nueva_j < N_COLS:
                casilla = tablero[nueva_i, nueva_j]
                
                if casilla == 0:  # Casilla vacía
                    # Si la casilla está vacía, podemos movernos ahí y seguir explorando
                    movimientos.append((origen, (nueva_i, nueva_j)))
                elif casilla * jugador < 0:  # Pieza enemiga
                    # Si encontramos una pieza enemiga, podemos capturarla pero no seguir
                    movimientos.append((origen, (nueva_i, nueva_j)))
                    break  # No puede seguir avanzando
                else:  # Pieza propia
                    # Si encontramos una pieza propia, bloqueamos esta dirección
                    break
                
                # Continuamos avanzando en la misma dirección
                nueva_i += di
                nueva_j += dj
        
        return movimientos
    
    def __movimientos_rey(
        self, 
        tablero: np.ndarray, 
        origen: Tuple[int, int], 
        jugador: int
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Calcula movimientos válidos para el rey (una casilla en cualquier dirección).
        """
        movimientos = []
        i, j = origen
        
        # Todas las direcciones posibles (8 direcciones)
        # El rey puede moverse una casilla en cualquiera de las 8 direcciones
        direcciones = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Comprobamos cada una de las 8 casillas adyacentes
        for di, dj in direcciones:
            nueva_i, nueva_j = i + di, j + dj
            
            # Verificamos que la nueva posición esté dentro del tablero
            if 0 <= nueva_i < N_FILAS and 0 <= nueva_j < N_COLS:
                casilla = tablero[nueva_i, nueva_j]
                
                # Puede moverse a casilla vacía o capturar pieza enemiga
                # El rey puede moverse si la casilla está vacía o tiene una pieza enemiga
                if casilla == 0 or casilla * jugador < 0:
                    movimientos.append((origen, (nueva_i, nueva_j)))
        
        return movimientos
    
    def __calcula_ganador(self) -> Union[int, None]:
        """
        Determina si hay un ganador en el estado actual del tablero.
        
        Condiciones de victoria:
        - Capturar el rey enemigo
        - Si no hay movimientos válidos para el oponente (jaque mate simplificado)
        - Empate si pasan 50 turnos sin captura
        
        :return: 1 si gana jugador 1, -1 si gana jugador 2, 0 si empate, None si continúa.
        """
        # Verificar si algún rey ha sido capturado
        # Buscamos ambos reyes en el tablero para ver si alguno fue capturado
        rey_j1_existe = False
        rey_j2_existe = False
        
        # Recorremos el tablero buscando los reyes
        for i in range(N_FILAS):
            for j in range(N_COLS):
                if self._tablero[i, j] == REY:
                    rey_j1_existe = True
                elif self._tablero[i, j] == -REY:
                    rey_j2_existe = True
        
        if not rey_j1_existe:  # Jugador 2 capturó el rey de jugador 1
            # Si no existe el rey del jugador 1, ha ganado el jugador 2
            self._fin = True
            return -1
        
        if not rey_j2_existe:  # Jugador 1 capturó el rey de jugador 2
            # Si no existe el rey del jugador 2, ha ganado el jugador 1
            self._fin = True
            return 1
        
        # Verificar si el jugador actual tiene movimientos válidos
        # Calculamos los movimientos posibles para ambos jugadores
        movimientos_j1 = self.__calcular_movimientos_validos(self._tablero, 1)
        movimientos_j2 = self.__calcular_movimientos_validos(self._tablero, -1)
        
        if len(movimientos_j1) == 0:  # Jugador 1 no puede mover
            # Si el jugador 1 no tiene movimientos válidos, pierde
            self._fin = True
            return -1
        
        if len(movimientos_j2) == 0:  # Jugador 2 no puede mover
            # Si el jugador 2 no tiene movimientos válidos, pierde
            self._fin = True
            return 1
        
        # Empate por repetición/estancamiento (50 turnos sin captura)
        # Si pasan demasiados turnos sin capturas, declaramos empate para evitar partidas infinitas
        if self._turnos_sin_captura >= 50:
            self._fin = True
            return 0
        
        # Partida continúa
        # Si no se cumple ninguna condición de finalización, la partida sigue
        self._fin = False
        return None
    
    def __actualiza_estado(
        self, 
        movimiento: Tuple[Tuple[int, int], Tuple[int, int]]
    ) -> None:
        """
        Actualiza el estado del tablero tras un movimiento.
        
        :param movimiento: Tupla (origen, destino).
        """
        # Extraemos las posiciones de origen y destino del movimiento
        origen, destino = movimiento
        pieza = self._tablero[origen]
        pieza_capturada = self._tablero[destino]
        
        # Realizar el movimiento
        # Movemos la pieza de origen a destino
        self._tablero[destino] = pieza
        self._tablero[origen] = 0
        
        # Actualizar contador de turnos sin captura
        # Reiniciamos el contador si hubo captura, sino lo incrementamos
        if pieza_capturada != 0:
            self._turnos_sin_captura = 0
        else:
            self._turnos_sin_captura += 1
        
        # Cambiar de jugador
        # Alternamos entre jugador 1 y jugador 2
        self._siguiente_jugador = -1 if self._siguiente_jugador == 1 else 1
        
        # Actualizar estado serializado
        # Guardamos la representación en string del nuevo estado del tablero
        self._estado = self._serializa_estado(self._tablero, self._n_filas, self._n_cols)
    
    def __recompensa(self) -> None:
        """
        Calcula y retropropaga las recompensas al final de la partida.
        
        Sistema de recompensas:
        - Victoria: 1.0
        - Derrota: 0.0
        - Empate: 0.5 (para ambos)
        """
        # Determinamos el resultado final de la partida
        resultado = self.__calcula_ganador()
        
        if resultado == 1:  # Victoria del jugador 1
            # El jugador 1 gana: recompensa máxima para él, mínima para el oponente
            self._jugador1.retropropaga_recompensa(1.0)
            if type(self._jugador2) == JugadorMiniChessMaq:
                self._jugador2.retropropaga_recompensa(0.0)
        
        elif resultado == -1:  # Victoria del jugador 2
            # El jugador 2 gana: recompensa mínima para jugador 1, máxima para jugador 2
            self._jugador1.retropropaga_recompensa(0.0)
            if type(self._jugador2) == JugadorMiniChessMaq:
                self._jugador2.retropropaga_recompensa(1.0)
        
        else:  # Empate
            # En caso de empate, ambos reciben una recompensa intermedia
            self._jugador1.retropropaga_recompensa(0.5)
            if type(self._jugador2) == JugadorMiniChessMaq:
                self._jugador2.retropropaga_recompensa(0.5)
    
    def __reset(self):
        """
        Resetea el juego para una nueva partida.
        """
        # Reiniciamos el tablero a su configuración inicial
        self._tablero = self.__inicializar_tablero()
        # Limpiamos las variables de control del juego
        self._estado = None
        self._fin = False
        self._siguiente_jugador = 1
        # Reiniciamos el contador de turnos sin captura
        self._turnos_sin_captura = 0
    
    def fit(self, rondas: int = 100) -> None:
        """
        Entrena los agentes mediante múltiples partidas.
        
        :param rondas: Número de partidas a jugar.
        """
        # Ejecutamos múltiples partidas para que los agentes aprendan
        for i in range(rondas):
            # Mostramos el progreso cada 100 partidas
            if i % 100 == 0:
                print(f"Ronda {i}/{rondas}")
            # Jugamos una partida completa
            self.jugar()
    
    def jugar(self) -> None:
        """
        Ejecuta una partida completa entre los dos jugadores.
        """
        # Determinar verbosidad según tipo de jugador 2
        # Solo mostramos el tablero si hay un jugador humano
        verbosidad = True 
        
        # Bucle principal del juego: continúa hasta que termine la partida
        while not self._fin:
            self.print_tablero(verbosidad)
            
            # Turno del jugador 1
            if self._siguiente_jugador == 1:
                # Calculamos todos los movimientos posibles para el jugador 1
                movimientos_validos = self.__calcular_movimientos_validos(
                    self._tablero, self._siguiente_jugador
                )
                
                if len(movimientos_validos) == 0:
                    # Si no hay movimientos válidos, la partida termina
                    break
                
                # El jugador 1 decide su acción
                accion_j1 = self._jugador1.decide_accion(
                    movimientos_validos, 
                    self._tablero, 
                    self._siguiente_jugador
                )
                
                # Ejecutamos el movimiento y guardamos el estado resultante
                self.__actualiza_estado(accion_j1)
                self._jugador1.guarda_estado(self._estado)
                
                # Verificar si la partida terminó
                # Comprobamos si este movimiento terminó la partida
                if self.__calcula_ganador() is not None:
                    break
            
            # Turno del jugador 2
            else:
                # Calculamos los movimientos válidos para el jugador 2
                movimientos_validos = self.__calcular_movimientos_validos(
                    self._tablero, self._siguiente_jugador
                )
                
                if len(movimientos_validos) == 0:
                    # Si no hay movimientos válidos, la partida termina
                    break
                
                # El jugador 2 decide su acción
                accion_j2 = self._jugador2.decide_accion(
                    movimientos_validos, 
                    self._tablero, 
                    self._siguiente_jugador
                )
                
                # Ejecutamos el movimiento
                self.__actualiza_estado(accion_j2)
                
                # Solo guardamos el estado si el jugador 2 es una máquina
                if type(self._jugador2) == JugadorMiniChessMaq:
                    self._jugador2.guarda_estado(self._estado)
                
                # Verificar si la partida terminó
                # Comprobamos si este movimiento terminó la partida
                if self.__calcula_ganador() is not None:
                    break
        
        # Fin de la partida: retropropagar recompensas e imprimir resultado
        # Mostramos el tablero final
        self.print_tablero(verbosidad)
        
        # Determinamos y mostramos el ganador si hay un humano jugando
        resultado = self.__calcula_ganador()
        if verbosidad:
            if resultado == 1:
                print(f"\n¡{self._jugador1.name} (Jugador 1) gana!")
            elif resultado == -1:
                print(f"\n¡{self._jugador2.name} (Jugador 2) gana!")
            else:
                print("\n¡Empate!")
        
        # Aplicamos las recompensas a los agentes según el resultado
        self.__recompensa()
        
        # Resetear para la siguiente partida
        # Preparamos todo para una nueva partida
        self._jugador1.reset()
        self._jugador2.reset()
        self.__reset()
    
    def print_tablero(self, verboso: bool = True) -> None:
        """
        Imprime el tablero actual con representación visual de las piezas.
        
        :param verboso: Si True, imprime el tablero.
        """
        # Si no queremos mostrar el tablero, salimos inmediatamente
        if not verboso:
            return
        
        # Imprimimos una línea separadora superior
        print("\n" + "=" * 33)
        
        # Mapeo de valores a símbolos
        # Definimos cómo representar cada pieza visualmente
        # Mayúsculas para jugador 1, minúsculas para jugador 2
        simbolos = {
            PEON: "P", TORRE: "T", REY: "R",
            -PEON: "p", -TORRE: "t", -REY: "r",
            0: " "
        }
        
        # Recorremos cada fila del tablero
        for i in range(N_FILAS):
            # Imprimimos el separador horizontal de cada fila
            print("  " + "-" * 29)
            # Comenzamos la línea con el número de fila
            linea = f"{i} |"
            
            # Añadimos cada pieza de la fila con su símbolo correspondiente
            for j in range(N_COLS):
                pieza = self._tablero[i, j]
                simbolo = simbolos.get(pieza, "?")
                linea += f" {simbolo} |"
            
            # Imprimimos la fila completa
            print(linea)
        
        # Imprimimos el borde inferior y los números de columna
        print("  " + "-" * 29)
        print("    0   1   2   3")
        print("=" * 33)


