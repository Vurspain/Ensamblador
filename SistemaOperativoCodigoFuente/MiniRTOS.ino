#include "DHT.h"
#include <Wire.h> 
#include <LiquidCrystal_I2C.h> // Librería de la pantalla

// --- DEFINICIÓN DE PINES ---
#define DHTPIN 2      
#define DHTTYPE DHT11 

const int fanPin = 8;
const int ledPin = 9;

// --- CONFIGURACIÓN DE OBJETOS ---
DHT dht(DHTPIN, DHTTYPE);

// DIRECCIÓN I2C: Generalmente es 0x27. Si no funciona, prueba 0x3F.
LiquidCrystal_I2C lcd(0x27, 16, 2);  

// --- ESTRUCTURA DE TAREAS ---
struct Task {
    void (*func)(); 
};

// Variables globales
float temperature = 0;
const float UMBRAL_TEMP = 30.0; 

// --- TAREA 1: Leer Temperatura ---
void task_readTemperature() {
    float t = dht.readTemperature();
    if (!isnan(t)) {
        temperature = t;
    }
}

// --- TAREA 2: Control del Ventilador ---
void task_controlFan() {
    if (temperature >= UMBRAL_TEMP) {
        digitalWrite(fanPin, HIGH); 
    } else {
        digitalWrite(fanPin, LOW);  
    }
}

// --- TAREA 3: Control del LED ---
void task_controlLED() {
    if (temperature >= UMBRAL_TEMP) {
        digitalWrite(ledPin, HIGH);
    } else {
        digitalWrite(ledPin, LOW);
    }
}

// --- TAREA 4 (NUEVA): Actualizar Pantalla LCD ---
void task_updateDisplay() {
    // Fila 0: Muestra la temperatura
    lcd.setCursor(0, 0); // Columna 0, Fila 0
    lcd.print("Temp: ");
    lcd.print(temperature);
    lcd.print(" C");

    // Fila 1: Muestra el estado del ventilador
    lcd.setCursor(0, 1); // Columna 0, Fila 1
    if (temperature >= UMBRAL_TEMP) {
        lcd.print("Estado: ALERTA! "); // Espacios para limpiar texto viejo
    } else {
        lcd.print("Estado: Normal  ");
    }
}

// --- SCHEDULER (Ahora con 4 tareas) ---
Task tasks[] = { 
    {task_readTemperature}, 
    {task_controlFan}, 
    {task_controlLED},
    {task_updateDisplay} // ¡Agregamos la pantalla a la lista!
};

int numTasks = sizeof(tasks) / sizeof(tasks[0]);

void setup() {
    pinMode(fanPin, OUTPUT);
    pinMode(ledPin, OUTPUT);
    
    Serial.begin(9600);
    dht.begin();
    
    // Iniciar la pantalla
    lcd.init();
    lcd.backlight(); // Prender la luz de fondo
    lcd.setCursor(0,0);
    lcd.print("Iniciando...");
    
    delay(1000); 
}

void loop() {
    for (int i = 0; i < numTasks; i++) {
        tasks[i].func();
    }
    
    // Debug en serial también, por si acaso
    Serial.print("Temp: ");
    Serial.println(temperature);

    delay(2000); 
}
