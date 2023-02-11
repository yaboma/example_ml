## Как локально разрабатывать
0) перейти в папку проекта и написать в командной строке ```docker build -t local_analyst . ```
1) ```docker run -d --name tochmash --user root -e GRANT_SUDO=yes -e RESTARTABLE=yes -v C:\Users\ivan\code\tochmash:/home/jovyan/work -p 8888:8888 local_analyst start-notebook.sh --NotebookApp.password='sha1:12834b604d46:85379ec0973745a8c3334aec2519f4c8b82dfeb9'``` при желании поменять пароль, текущий - tochmash
2) При желании подключиться через vscode

## Описание текущего эксперимента с предсказанием масс по биениям и предыдущим испытаниям (каждое испытание == новый ротор)
1) Предварительно укладываем данные в 01_clear_data.ipynb
2) Вычисляем признаки для каждого текущего испытания по биениям в 02_EDA.ipynb
3) Вычисляем признаки предыдущего испытания и разности с текущим в 02_EDA_add_shifted_and_windowed_features.ipynb (здесь сейчас еще фильтруются неполные месяцы - это нужно перенести в самое начало)
4) обучаем модель в 03_EDA_learning_with_previous_test.ipynb
Текущий результат: Общий MiultiRMSE для низкочастотных масс - 35.894<br>
Для 0 испытания - 72.4218<br>
Для следующих - 13.745<br>
В среднем RMSE ~ 20 на каждый из видов масс

## Запуск минимального сервера
0) в корне: ```poetry export -f requirements.txt --output app/back/requirements.txt --without-hashes``` - чтобы получить корректные версии как в ноутбуке
1) ```cd app/```
2) ```docker-compose build ```
3) ```docker-compose up -d```
4) перейти на http://localhost:8501/ для ui - можно добавлять файлы
5) перейти на http://localhost:8000/docs для описания текущего API
6) можно подключиться к postgresql базе данных, в которой сейчас хранятся результаты работы сервиса (креденшиалы нужно задать при запуске сервиса, см пункт про миграции)

## Текущие доработки сервера
1) ~~Добавление БД вешает сервис, нужно добавить нормальновключая миграции~~
2) ~~Есть проблема с асинхронностью при вычислении предикта (или нет проблемы, здесь нужно еще поразбираться)~~ обернуто в threadPool, теперь вроде как проблемы точно нет
3) ~~логгирование отсутствует~~ можно лучше

## Запуск миграции 
0) ```cd app ```, локально определить переменные окружения (лучше, через .env файл в app, не забыв добавить при необходимости в .gitignore)
<pre>
SUPERUSER_PASS=# Пароль для суперпользователя БД
DB_NAME=# Название БД, в кторую запишутся результаты
DB_USER=# имя пользователя, от имени которого в этой БД можно хозяйничать
DB_PASS=# пароль пользователя
DB_HOST=# имя контейнера с БД
</pre>
1) Создать БД внутри нужного докер тома, нпример так
    1) ```docker-compose run --rm mas_calculator psql -h storage -p 5432 -U postgres ``` и ввеси пароль от суперпользователя
    2) Далее нужно создать БД согласно вашим переменным окружения - ToDo, было бы неплохо это делать автоматом через дополнительный sh скрипт (это уже сделано в тестах, можно вынести оттуда при возможности).
    ~~~~sql
        create database <$DB_NAME>;
        create user <$DB_USER> with encrypted password '<$DB_PASS>';
        grant all privileges on database <$DB_NAME> to <$DB_USER>;
    ~~~~

    3) <code>alembic init db/migrations</code> - если у вас нет миграций в db/migrations
2) ```docker-compose run --rm  mas_calculator alembic revision --autogenerate -m "Added required tables"```
3) ```docker-compose run --rm  mas_calculator alembic upgrade head ```

## Тесты
1) <code>cd app</code>
2) <code>docker-compose run --rm mas_calculator py.test --setup-show --asyncio-mode=strict</code> - для бека
3) <code>docker-compose run --rm front py.test --setup-show  - для фронта</code>