//파이썬연동, servo1, servo2, delay 입력
#include <Servo.h>  //servo모터 제어

#define MSEC  1
#define SEC 1000

enum RobotState { READY, CATCH, ARMMOVE, UNCATCH };

Servo servo1, servo2, servo3, servo4, servo5;
int targetPos1, targetPos2, targetPos3; //목표각도
int jodo = 30;
int score, Delay = 180;

const int startPos1 = 40;
const int startPos2 = 110;
const int startPos3 = 110;
const int startPos4 = 0;

void setup()
{
  Serial.begin(9600);

  servo1.attach(2); //어깨
  delay(100);
  servo2.attach(3); //팔꿈치
  delay(100);
  servo3.attach(4); //손목
  delay(100);
  servo4.attach(5); //손가락

  //  delay(100);
  startRobot();
}
RobotState robotState;
bool runTimeReady = false, runTimeArmMove = false, runTimeUncatch = false, runCheckJodo = false;
long oldTimeReady = 0, oldTimeArmMove = 0, oldTimeUncatch = 0, oldTimeSerial = 0, oldTimeCheckJodo = 0;
void loop()
{
  if (runTimeReady)
  {
    if (millis() - oldTimeReady > 2 * SEC)
    {
      changeCatchState();
      runTimeReady = false;
    }
  }
  if (runTimeArmMove)
  {
    if (millis() - oldTimeArmMove > 2 * SEC)
    {
      changeArmMoveState();
      runTimeArmMove = false;
    }
  }
  if (runTimeUncatch)
  {
    if (millis() - oldTimeUncatch > Delay * MSEC)
    {
      changeUncatchState();
      runTimeUncatch = false;
    }
  }
  if (runCheckJodo)
  {
    CheckJodo();

    if (score > 0 || millis() - oldTimeCheckJodo > 3 * SEC)
    {

      changeReadyState();
      runCheckJodo = false;
    }
  }
  //  if( millis()-oldTimeSerial > 100*MSEC)
  //  {
  //    switch(robotState)
  //    {
  //      case READY:
  //      Serial.println("State : READY");
  //      break;
  //      case CATCH:
  //      Serial.println("State : CATCH");
  //      break;
  //      case ARMMOVE:
  //      Serial.println("State : ARMMOVE");
  //      break;
  //      case UNCATCH:
  //      Serial.println("State : UNCATCH");
  //      break;
  //    }
  //    oldTimeSerial = millis();
  //  }
}
void startRobot()
{
  changeReadyState();
}
void changeReadyState()
{
  robotState = READY;

  servo1.write(startPos1);
  servo2.write(startPos2);
  servo3.write(startPos3);
  servo4.write(startPos4);

  runTimeReady = true;
  oldTimeReady = millis();
}
void changeCatchState()
{
  robotState = CATCH;

  servo4.write(70);

  runTimeArmMove = true;
  oldTimeArmMove = millis();
}
void changeArmMoveState()
{
  robotState = ARMMOVE;

//  targetPos1 = random(95, 160);
//  targetPos2 = random(50, 80);
  
//  targetPos1 = 90;
//  targetPos2 = 50;
  targetPos3 = 18;
  //servo1범위 21~167  160이 앞
  while(true)
  {
    if(Serial.available())
    {
      targetPos1=Serial.parseInt();
      break;
    }
  }
  Serial.println(targetPos1);
   
    while(true)
  {
    if(Serial.available())
    {
      targetPos2=Serial.parseInt();
      break;
    }
  }
  Serial.println(targetPos2);
  servo1.write(targetPos1);
  servo2.write(targetPos2);
  servo3.write(70);


  runTimeUncatch = true;
  oldTimeUncatch = millis();
}

void changeUncatchState()
{
  robotState = UNCATCH;

  servo4.write(0);

  runCheckJodo = true;

  oldTimeCheckJodo = millis();
}
void CheckJodo()
{
  int val[10];
  val[0] = analogRead(A0),    val[1] = analogRead(A1);
  val[2] = analogRead(A2),    val[3] = analogRead(A3);
  val[4] = analogRead(A4),    val[5] = analogRead(A5);
  val[6] = analogRead(A6),    val[7] = analogRead(A7);
  val[8] = analogRead(A8),    val[9] = analogRead(A9);

  if (val[0] < jodo || val[1] < jodo)
    score = 1;
  else if (val[2] < jodo || val[3] < jodo)
    score = 2;
  else if (val[4] < jodo || val[5] < jodo)
    score = 3;
  else if (val[6] < jodo || val[7] < jodo)
    score = 4;
  else if (val[8] < jodo || val[9] < jodo)
    score = 5;
  else
    score = 0;
}
