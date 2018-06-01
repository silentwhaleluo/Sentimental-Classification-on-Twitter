proc format library=work;
	value Sentimentfmt 1='Positive' 0='Negative';

FILENAME REFFILE '/folders/myfolders/BIA652_train_1000_DR.csv';

PROC IMPORT DATAFILE=REFFILE replace
	DBMS=CSV
	OUT=WORK.data;
	GETNAMES=YES;
RUN;

proc logistic data=WORK.data;
	model sentiment=X1-X369 / selection=stepwise sle=0.1 sls=0.1 STB Technique=fisher outroc=roc1;
	format sentiment sentimentfmt.;
run;

proc discrim data=WORK.data pool=yes anova manova listerr crosslisterr;
	class sentiment;
	format sentiment sentimentfmt.;
	var X1-X369;
run;