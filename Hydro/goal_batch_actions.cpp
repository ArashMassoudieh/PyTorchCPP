// Goal-driven Batch action: invokes the same bounded backend as the CLI.
#include <QAction>
#include <QApplication>
#include <QCoreApplication>
#include <QDialog>
#include <QDir>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QMainWindow>
#include <QMenuBar>
#include <QMenu>
#include <QMessageBox>
#include <QProcess>
#include <QPushButton>
#include <QTextEdit>
#include <QTimer>
#include <QVBoxLayout>
#include <QFile>
#include <QUuid>
#include <memory>
namespace {
QString rootPath(){
 for(const QString& start:{QDir::currentPath(),QCoreApplication::applicationDirPath()}){
  QDir d(start);for(int n=0;n<10;++n){if(QFileInfo::exists(d.filePath("HydroPINN.pro")))return d.absolutePath();if(!d.cdUp())break;}
 }return {};
}
void openGoalBatch(QMainWindow* window){
 const QString repo=rootPath(),script=repo+"/Hydro/goal_batch/controller.py";
 if(repo.isEmpty()||!QFileInfo::exists(script)){QMessageBox::critical(window,"Goal-driven training","Cannot locate the shared batch controller.");return;}
 auto* dialog=new QDialog(window);dialog->setWindowTitle("Goal-driven training — bounded development search");dialog->resize(820,600);
 auto* layout=new QVBoxLayout(dialog);auto* explanation=new QLabel("Choose a verified request containing a finite candidate inventory and resource limits.\nNSE measures agreement; R² is squared Pearson correlation. Targets are goals, not guarantees.\nOnly cropped training/validation packages are accepted; reserved data and test-based selection are blocked.",dialog);explanation->setWordWrap(true);layout->addWidget(explanation);
 auto* form=new QFormLayout;auto* nse=new QDoubleSpinBox(dialog);auto* r2=new QDoubleSpinBox(dialog);
 for(auto* spin:{nse,r2}){spin->setRange(0,1);spin->setDecimals(3);spin->setSingleStep(.05);spin->setValue(.7);}
 form->addRow("Minimum validation NSE",nse);form->addRow("Minimum validation R²",r2);layout->addLayout(form);
 auto* start=new QPushButton("New bounded run from request…",dialog);auto* resume=new QPushButton("Resume verified run…",dialog);auto* cancel=new QPushButton("Cancel current run",dialog);cancel->setEnabled(false);
 layout->addWidget(start);layout->addWidget(resume);layout->addWidget(cancel);auto* output=new QTextEdit(dialog);output->setReadOnly(true);layout->addWidget(output);
 auto* process=new QProcess(dialog);process->setProcessChannelMode(QProcess::MergedChannels);process->setWorkingDirectory(repo);
 auto runRoot=std::make_shared<QString>();auto preparing=std::make_shared<bool>(false);auto cancelled=std::make_shared<bool>(false);
 auto busy=[=](bool b){start->setEnabled(!b);resume->setEnabled(!b);nse->setEnabled(!b);r2->setEnabled(!b);cancel->setEnabled(b);};
 QObject::connect(process,&QProcess::readyReadStandardOutput,dialog,[=](){output->append(QString::fromUtf8(process->readAllStandardOutput()));});
 QObject::connect(process,qOverload<int,QProcess::ExitStatus>(&QProcess::finished),dialog,[=](int code,QProcess::ExitStatus status){
  if(*preparing&&code==0&&status==QProcess::NormalExit){*preparing=false;if(*cancelled){QFile marker(*runRoot+"/CANCEL");if(marker.open(QIODevice::WriteOnly|QIODevice::NewOnly))marker.write("{}\n");}process->start("python3",{script,"run",*runRoot});return;}
  *preparing=false;busy(false);output->append(code==0&&status==QProcess::NormalExit?"Controller finished. Read report.json and state.json for the goal decision.":"Stopped with an error. Existing claims remain charged; no failed fit is automatically retried.");
 });
 QObject::connect(process,&QProcess::errorOccurred,dialog,[=](QProcess::ProcessError){output->append(process->errorString());busy(false);});
 QObject::connect(start,&QPushButton::clicked,dialog,[=](){
  const QString request=QFileDialog::getOpenFileName(dialog,"Verified candidate request",repo,"JSON (*.json)");if(request.isEmpty())return;
  const QString parent=QFileDialog::getExistingDirectory(dialog,"Parent directory for a new run");if(parent.isEmpty())return;
  const QString id=QUuid::createUuid().toString(QUuid::WithoutBraces);*runRoot=parent+"/goal_batch_"+id;
  output->append("Run: "+*runRoot);busy(true);*cancelled=false;*preparing=true;process->start("python3",{script,"prepare",*runRoot,"--request",request,"--nse",QString::number(nse->value(),'g',17),"--r-squared",QString::number(r2->value(),'g',17)});
 });
 QObject::connect(resume,&QPushButton::clicked,dialog,[=](){
  const QString dir=QFileDialog::getExistingDirectory(dialog,"Existing run; verified jobs are not repeated");if(dir.isEmpty())return;*runRoot=dir;busy(true);*cancelled=false;*preparing=false;output->append("Resuming frozen plan: "+dir+" (its stored goals apply).");process->start("python3",{script,"run",dir});
 });
 QObject::connect(cancel,&QPushButton::clicked,dialog,[=](){
  *cancelled=true;
  if(runRoot->isEmpty()){return;}
  QFile marker(*runRoot+"/CANCEL");if(marker.open(QIODevice::WriteOnly|QIODevice::NewOnly)){marker.write("{}\n");marker.close();output->append("Cancellation requested; interrupted fits cannot be retried.");}
 });
 // Closing must not kill the controller and orphan its managed process group.
 QObject::connect(dialog,&QDialog::rejected,window,[=](){if(process->state()!=QProcess::NotRunning){*cancelled=true;QFile marker(*runRoot+"/CANCEL");if(marker.open(QIODevice::WriteOnly|QIODevice::NewOnly))marker.write("{}\n");}});
 dialog->show();
}
void installGoalBatch(){
 QMainWindow* window=nullptr;for(auto* w:QApplication::topLevelWidgets())if((window=qobject_cast<QMainWindow*>(w)))break;
 if(!window){QTimer::singleShot(100,installGoalBatch);return;}
 QMenu* menu=nullptr;for(auto* a:window->menuBar()->actions())if(QString(a->text()).remove('&')=="Batch")menu=a->menu();
 if(!menu){QTimer::singleShot(100,installGoalBatch);return;}
 if(menu->findChild<QAction*>("HydroGoalBatchAction"))return;
 auto* action=new QAction("Goal-driven training…",menu);action->setObjectName("HydroGoalBatchAction");menu->addAction(action);QObject::connect(action,&QAction::triggered,window,[=](){openGoalBatch(window);});
}
void scheduleGoalBatch(){QTimer::singleShot(0,installGoalBatch);}
}
Q_COREAPP_STARTUP_FUNCTION(scheduleGoalBatch)
