[Setup]
; Имя программы в списке установленных приложений Windows
AppName=Chembot AI Agent
AppVersion=1.0
; Путь установки по умолчанию: C:\Program Files\Chembot
DefaultDirName={autopf}\Chembot
DefaultGroupName=Chembot
; Иконка для раздела "Удаление программ" в Панели управления
UninstallDisplayIcon={app}\chembot.exe
Compression=lzma
SolidCompression=yes
; Иконка самого файла инсталлятора (setup.exe)[cite: 1]
SetupIconFile=assets\app_icon.ico 
; Куда сохранится готовый установщик[cite: 1]
OutputDir=userdocs:Inno Setup Outputs

[Tasks]
; Галочка "Создать ярлык на рабочем столе"[cite: 1]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Если в dist лежит папка "chembot", то пишем "dist\chembot\*"
Source: "dist\chembot\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; Ярлыки в меню Пуск и на Рабочем столе[cite: 1]
Name: "{group}\Chembot AI"; Filename: "{app}\chembot.exe"
Name: "{autodesktop}\Chembot AI"; Filename: "{app}\chembot.exe"; Tasks: desktopicon

[Run]
; Запуск программы сразу после завершения установки[cite: 1]
Filename: "{app}\chembot.exe"; Description: "{cm:LaunchProgram,Chembot AI}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Полная очистка: удаляет базу данных и логи из AppData при деинсталляции[cite: 1]
; Убедись, что в коде Python ты используешь именно эту папку для сохранения данных
Type: filesandordirs; Name: "{userappdata}\AIDesktopAgent"