; 联安智盾 Windows 安装器（Inno Setup 6）。
; 编译：
;   ISCC /DAppVersion=1.0.0 packaging\liananzhidun.iss
; 输入目录由 packaging/build_portable.py 生成（build\stage）。

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif

#define AppName "联安智盾"
#define AppNameEn "LianAnZhiDun"
#define AppPublisher "JayJoneCoder"
#define StageDir "..\build\stage"

[Setup]
AppId={{E3B1A7C5-6D42-4F1B-9A3E-2C8D5F7B0A64}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright=AGPL-3.0-or-later
DefaultDirName={localappdata}\Programs\{#AppNameEn}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
AllowNoIcons=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename={#AppNameEn}-{#AppVersion}-win64-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
AppMutex=LianAnZhiDunPlatform
SetupIconFile=assets\app.ico
UninstallDisplayIcon={app}\app.ico
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoProductName={#AppName}
VersionInfoDescription={#AppName} 安装程序
CloseApplications=no
RestartApplications=no

[Languages]
Name: "default"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加任务:"; Flags: unchecked

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "assets\app.ico"; DestDir: "{app}"; Flags: ignoreversion
#if FileExists(SourcePath + "vendor\vc_redist.x64.exe")
Source: "vendor\vc_redist.x64.exe"; DestDir: "{tmp}"; Flags: deleteafterinstall; Check: not VCRedistInstalled
#endif

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-B ""{app}\desktop\start_gui.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{group}\运行环境自检"; Filename: "{cmd}"; Parameters: "/k ""{app}\python\python.exe"" -B -m desktop.launcher --diagnose"; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"
Name: "{group}\卸载 {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\python\pythonw.exe"; Parameters: "-B ""{app}\desktop\start_gui.py"""; WorkingDir: "{app}"; IconFilename: "{app}\app.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-B ""{app}\desktop\start_gui.py"""; WorkingDir: "{app}"; Description: "立即启动{#AppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Runtime-generated __pycache__ is not tracked by the installer.
Type: filesandordirs; Name: "{app}"

[Messages]
SetupAppTitle=安装程序
SetupWindowTitle=安装程序 - %1
UninstallAppTitle=卸载
UninstallAppFullTitle=卸载 %1
ButtonBack=< 上一步(&B)
ButtonNext=下一步(&N) >
ButtonInstall=安装(&I)
ButtonOK=确定
ButtonCancel=取消
ButtonYes=是(&Y)
ButtonNo=否(&N)
ButtonFinish=完成(&F)
ButtonBrowse=浏览(&B)...
WelcomeLabel1=欢迎使用 [name] 安装向导
WelcomeLabel2=安装程序将把 [name/ver] 安装到您的电脑。%n%n建议在继续之前关闭其它应用程序。
WizardSelectDir=选择安装位置
SelectDirDesc=联安智盾将安装到哪个文件夹？
SelectDirLabel3=安装程序将把 [name] 安装到以下文件夹。
SelectDirBrowseLabel=点击“下一步”继续；如需选择其它文件夹，请点击“浏览”。
DiskSpaceMBLabel=至少需要 [mb] MB 的可用磁盘空间。
WizardReady=准备安装
ReadyLabel1=安装程序已准备好在您的电脑上安装 [name]。
ReadyLabel2a=点击“安装”开始安装；如需检查或更改设置，请点击“上一步”。
ReadyLabel2b=点击“安装”开始安装。
WizardInstalling=正在安装
InstallingLabel=正在安装 [name]，请稍候…
FinishedHeadingLabel=[name] 安装完成
FinishedLabel=安装程序已完成 [name] 的安装。%n%n现在可以启动平台；首次启动会打开浏览器，请选择下载源和 CPU/GPU 版本。
FinishedLabelNoIcons=安装程序已完成 [name] 的安装。
ClickFinish=点击“完成”退出安装程序。
PreparingDesc=安装程序正在准备安装 [name]，请稍候…
ExitSetupTitle=退出安装
ExitSetupMessage=安装尚未完成。如果现在退出，平台将不会被安装。%n%n确定要退出吗？
ConfirmUninstall=确定要完全移除 %1 及其所有组件吗？
UninstalledAll=%1 已从您的电脑中成功移除。
UninstalledMost=%1 卸载完成。%n%n部分文件需要手动删除。

[Code]
function VCRedistInstalled: Boolean;
var
  Installed: Cardinal;
begin
  Result := RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Installed', Installed) and (Installed = 1);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if (CurStep = ssPostInstall) and (not VCRedistInstalled) then
  begin
    if FileExists(ExpandConstant('{tmp}\vc_redist.x64.exe')) then
    begin
      Log('Installing Microsoft Visual C++ 2015-2022 Redistributable');
      Exec(ExpandConstant('{tmp}\vc_redist.x64.exe'), '/install /quiet /norestart', '',
           SW_HIDE, ewWaitUntilTerminated, ResultCode);
      Log('VC++ redist exit code: ' + IntToStr(ResultCode));
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\LianAnZhiDun');
    if DirExists(DataDir) and (not UninstallSilent) then
    begin
      if MsgBox('是否同时删除账号、配置、实验记录和已下载的 PyTorch 运行环境？' + #13#10 + #13#10 + DataDir,
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
