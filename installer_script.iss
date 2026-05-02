[Setup]
AppName=AI Desktop Agent
AppVersion=1.0
DefaultDirName={autopf}\AIDesktopAgent
DefaultGroupName=AI Desktop Agent
UninstallDisplayIcon={app}\AIDesktopAgent.exe
Compression=lzma
SolidCompression=yes
; Указываем путь к иконке установщика
SetupIconFile=assets\app_icon.ico 
OutputDir=userdocs:Inno Setup Outputs

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Копируем всё содержимое папки dist после сборки flet pack
Source: "dist\AIDesktopAgent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AI Desktop Agent"; Filename: "{app}\AIDesktopAgent.exe"
Name: "{autodesktop}\AI Desktop Agent"; Filename: "{app}\AIDesktopAgent.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AIDesktopAgent.exe"; Description: "{cm:LaunchProgram,AI Desktop Agent}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Опционально: удалять данные из AppData при деинсталляции
Type: filesandordirs; Name: "{userappdata}\AIDesktopAgent"