// MotionPlay - ESP32 + MPU6050 controller
//
// Streams IMU + button over UDP (50 Hz) and reacts to feedback commands from the game.
// Every add-on is OPTIONAL: set its flag to 1 only after you have wired it and installed its library.
// With every flag at 0 it only streams IMU + button data (no feedback).
//
// ---------------- WIRING (all I2C parts share the SAME two wires) ----------------
//  MPU6050  VCC->3V3  GND->GND  SDA->GPIO21  SCL->GPIO22
//  OLED     VCC->3V3  GND->GND  SDA->GPIO21  SCL->GPIO22   (SSD1306 128x64 I2C, address 0x3C)
//  Button   one leg->GPIO27, other leg->GND                (internal pull-up used)
//  Buzzer   (ACTIVE buzzer) + -> GPIO26, - -> GND
//  Vibration motor: use a motor MODULE (has its own driver) IN->GPIO25, VCC->3V3, GND->GND
//                   (a bare motor needs NPN transistor + 1k base resistor + diode - never drive it from a pin directly)
//  RGB LED  WS2812 / NeoPixel  DIN->GPIO4 (via 330 ohm), VCC->3V3 (or 5V/VIN), GND->GND
//
// Libraries (Arduino IDE -> Library Manager), only for the flags you enable:
//  USE_OLED : "Adafruit SSD1306" + "Adafruit GFX Library"
//  USE_RGB  : "Adafruit NeoPixel"
//
// Packet (ASCII CSV, 50 Hz):  pid,seq,t_ms,ax,ay,az,gx,gy,gz,btn     (accel in g, gyro in deg/s)
// Commands from the laptop (UDP port 4211):  H hit | M miss | X bad-orb hit | G gesture | S,score,lives,level,combo | E,score

#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>

// ---------- EDIT THESE ----------
const char* WIFI_SSID = "Nothing Phone (2a)";
const char* WIFI_PASS = "hellokitty";
const char* LAPTOP_IP = "10.164.233.16";      // `ipconfig` on the laptop (IPv4 address)
#define PLAYER_ID   1                        // 1 = first controller, 2 = second controller (versus mode)

#define USE_BUTTON  0
#define USE_BUZZER  1
#define BUZZER_ACTIVE_LOW 1                  // 3-pin module (VCC, I/O, GND) usually sounds when I/O is LOW. If it is silent/always on, flip to 0
#define USE_MOTOR   0
#define USE_OLED    0
#define USE_RGB     0                        // NeoPixel / WS2812 (3 pins) - keep 0
#define USE_LED     1                        // plain 2-leg LED on GPIO4 through a 220-330 ohm resistor
// --------------------------------

const uint16_t UDP_PORT = 4210;              // controller -> laptop
const uint16_t CMD_PORT = 4211;              // laptop -> controller

const int PIN_BTN = 27, PIN_BUZZER = 26, PIN_MOTOR = 25, PIN_RGB = 4;
const int RGB_COUNT = 1;                     // number of NeoPixels

const uint8_t MPU_ADDR = 0x68;
const uint32_t SAMPLE_PERIOD_MS = 20;        // 50 Hz
const float ACC_LSB = 16384.0;               // +-2 g
const float GYR_LSB = 131.0;                 // +-250 deg/s   (a hard flick can clip at 250: raise to +-500 below if needed)

#if USE_OLED
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
Adafruit_SSD1306 oled(128, 64, &Wire, -1);
bool oledOK = false;
#endif
#if USE_RGB
#include <Adafruit_NeoPixel.h>
Adafruit_NeoPixel px(RGB_COUNT, PIN_RGB, NEO_GRB + NEO_KHZ800);
#endif

WiFiUDP udp, rxUdp;
uint32_t seq = 0, lastSample = 0;
uint32_t buzzOff = 0, motorOff = 0, ledOff = 0, lastOled = 0;
int gScore = 0, gLives = 0, gLevel = 1, gCombo = 0;
bool gOver = false, gDirty = true;

// ------------------------------------------------------------ MPU6050
void mpuWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

bool mpuRead(int16_t* v) {                   // v[0..2]=acc, v[3]=temp, v[4..6]=gyro
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)MPU_ADDR, 14, (int)true) != 14) return false;
  for (int i = 0; i < 7; i++) {
    int16_t hi = Wire.read();
    v[i] = (hi << 8) | Wire.read();
  }
  return true;
}

// ------------------------------------------------------------ feedback helpers (all non-blocking)
void setLed(uint8_t r, uint8_t g, uint8_t b, uint32_t ms) {
#if USE_LED
  bool on = !(ms == 0 && r == 0 && g == 0);      // idle colour (0,0,40,0) = LED off
  digitalWrite(PIN_RGB, on ? HIGH : LOW);
  ledOff = ms ? millis() + ms : 0;
#endif
#if USE_RGB
  for (int i = 0; i < RGB_COUNT; i++) px.setPixelColor(i, px.Color(r, g, b));
  px.show();
  ledOff = ms ? millis() + ms : 0;
#endif
}
void buzz(uint32_t ms) {
#if USE_BUZZER
  digitalWrite(PIN_BUZZER, BUZZER_ACTIVE_LOW ? LOW : HIGH);
  buzzOff = millis() + ms;
#endif
}
void vibrate(uint32_t ms) {
#if USE_MOTOR
  digitalWrite(PIN_MOTOR, HIGH);
  motorOff = millis() + ms;
#endif
}

void feedback(char c) {
  switch (c) {
    case 'H': buzz(40);  vibrate(40);  setLed(0, 255, 80, 150);   break;   // caught an orb
    case 'M': vibrate(60);             setLed(255, 150, 0, 120);   break;   // missed an orb
    case 'X': buzz(250); vibrate(300); setLed(255, 0, 0, 400);    break;   // red orb hit (lost a life)
    case 'G': buzz(25);                setLed(60, 120, 255, 200);  break;   // gesture used
  }
}

void showStatus() {
#if USE_OLED
  if (!oledOK) return;
  char b[32];
  oled.clearDisplay();
  oled.setTextColor(SSD1306_WHITE);
  oled.setTextSize(1);
  oled.setCursor(0, 0);
  snprintf(b, sizeof(b), "P%d   LV %d", PLAYER_ID, gLevel);
  oled.print(b);
  oled.setCursor(0, 14);
  oled.setTextSize(3);
  snprintf(b, sizeof(b), "%d", gScore);
  oled.print(b);
  oled.setTextSize(1);
  oled.setCursor(0, 46);
  if (gOver) {
    oled.print("GAME OVER");
  } else {
    snprintf(b, sizeof(b), "combo %d", gCombo);
    oled.print(b);
    for (int i = 0; i < gLives && i < 8; i++) oled.fillCircle(8 + i * 15, 58, 4, SSD1306_WHITE);
  }
  oled.display();
#endif
}

void handleCmd(char* b) {
  switch (b[0]) {
    case 'H': case 'M': case 'X': case 'G': feedback(b[0]); break;
    case 'S': {
      int s, l, v, c;
      if (sscanf(b + 2, "%d,%d,%d,%d", &s, &l, &v, &c) == 4) {
        gScore = s; gLives = l; gLevel = v; gCombo = c; gOver = false; gDirty = true;
      }
      break;
    }
    case 'E':
      gOver = true; gDirty = true;
      sscanf(b + 2, "%d", &gScore);
      setLed(160, 60, 255, 0);
      buzz(400);
      break;
  }
}

// ------------------------------------------------------------ setup / loop
void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  Wire.setClock(400000);
  mpuWrite(0x6B, 0x00);   // wake up
  mpuWrite(0x1A, 0x03);   // DLPF ~44 Hz
  mpuWrite(0x1B, 0x00);   // gyro +-250 deg/s   (0x08 = +-500, 0x10 = +-1000; then change GYR_LSB to 65.5 / 32.8)
  mpuWrite(0x1C, 0x00);   // accel +-2 g

#if USE_BUTTON
  pinMode(PIN_BTN, INPUT_PULLUP);
#endif
#if USE_BUZZER
  pinMode(PIN_BUZZER, OUTPUT); digitalWrite(PIN_BUZZER, BUZZER_ACTIVE_LOW ? HIGH : LOW);
#endif
#if USE_MOTOR
  pinMode(PIN_MOTOR, OUTPUT); digitalWrite(PIN_MOTOR, LOW);
#endif
#if USE_LED
  pinMode(PIN_RGB, OUTPUT); digitalWrite(PIN_RGB, LOW);
  setLed(255, 120, 0, 0);                    // on = connecting to WiFi
#endif
#if USE_RGB
  px.begin(); px.setBrightness(60);
  setLed(255, 120, 0, 0);                    // orange = connecting to WiFi
#endif
#if USE_OLED
  oledOK = oled.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  if (oledOK) { oled.clearDisplay(); oled.setTextColor(SSD1306_WHITE); oled.setCursor(0, 0); oled.print("MotionPlay\nconnecting..."); oled.display(); }
#endif

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);                      // lower latency
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(300); Serial.print("."); }
  Serial.print("\nConnected, ESP32 IP: ");
  Serial.println(WiFi.localIP());
  rxUdp.begin(CMD_PORT);
  setLed(0, 0, 40, 0);                       // dim blue = connected, waiting for the game

#if USE_OLED
  if (oledOK) { oled.clearDisplay(); oled.setCursor(0, 0); oled.print("connected\n"); oled.print(WiFi.localIP()); oled.print("\nwaiting for game"); oled.display(); }
#endif
}

void loop() {
  uint32_t now = millis();

  // ---- switch actuators off when their time is up
#if USE_BUZZER
  if (buzzOff && now > buzzOff) { digitalWrite(PIN_BUZZER, BUZZER_ACTIVE_LOW ? HIGH : LOW); buzzOff = 0; }
#endif
#if USE_MOTOR
  if (motorOff && now > motorOff) { digitalWrite(PIN_MOTOR, LOW); motorOff = 0; }
#endif
#if USE_RGB || USE_LED
  if (ledOff && now > ledOff) { setLed(0, 0, 40, 0); }
#endif

  // ---- feedback commands from the laptop
  int n = rxUdp.parsePacket();
  if (n > 0) {
    char b[64];
    int len = rxUdp.read(b, sizeof(b) - 1);
    if (len > 0) { b[len] = 0; handleCmd(b); }
  }
#if USE_OLED
  if (gDirty && now - lastOled > 500) { showStatus(); gDirty = false; lastOled = now; }   // OLED redraw takes ~25 ms: keep it rare
#endif

  // ---- sample + send at 50 Hz
  if (now - lastSample < SAMPLE_PERIOD_MS) return;
  lastSample += SAMPLE_PERIOD_MS;
  if (now - lastSample > 200) lastSample = now;   // resync after a stall

  int16_t r[7];
  if (!mpuRead(r)) {
    static uint32_t lastMsg = 0;
    if (now - lastMsg > 3000) {                // print rarely, so the IP stays visible
      lastMsg = now;
      Serial.print("MPU read failed (check wiring/solder). ESP32 IP: ");
      Serial.println(WiFi.localIP());
    }
    return;
  }

  int btn = 0;
#if USE_BUTTON
  btn = (digitalRead(PIN_BTN) == LOW) ? 1 : 0;
#endif

  char buf[128];
  int len = snprintf(buf, sizeof(buf), "%d,%lu,%lu,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%d",
                     PLAYER_ID, (unsigned long)seq++, (unsigned long)now,
                     r[0] / ACC_LSB, r[1] / ACC_LSB, r[2] / ACC_LSB,
                     r[4] / GYR_LSB, r[5] / GYR_LSB, r[6] / GYR_LSB, btn);
  udp.beginPacket(LAPTOP_IP, UDP_PORT);
  udp.write((const uint8_t*)buf, len);
  udp.endPacket();
}
