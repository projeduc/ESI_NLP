% Designed by: Abdelkrime Aries
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{aakpract}[2025/02/22 v1.0 AAK Practicals class]

%===============================================================
% Options
%===============================================================


%\newif\ifshort % short vrsion
%\shortfalse % by default it is long
%\newif\ifinfo % information
%\infofalse % by default no information
%\newif\ifsmp % simpl
%\smpfalse % by default no information
%
%\DeclareOption{short}{\shorttrue}
%\DeclareOption{info}{\infotrue}
%\DeclareOption{smp}{\smptrue}
%
%\ProcessOptions\relax

\LoadClass[12pt, a4paper]{article}


%===============================================================
% Packages
%===============================================================

\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amssymb}
\usepackage[centertags]{amsmath}
\usepackage{multirow}
\usepackage{hyperref}
%\usepackage{natbib}
\usepackage[explicit]{titlesec}
\usepackage{tcolorbox}
\usepackage{times}
\usepackage{titlecaps}
\usepackage{tabularx}
\usepackage{longtable}
\usepackage[table]{xcolor}
\usepackage{etoolbox} %list variables, if else
\usepackage{tabularray}
\usepackage{fancyhdr}
\usepackage[natbibapa]{apacite}
\usepackage{enumitem}
\usepackage{url}
\usepackage{relsize}
\usepackage{nicematrix, tikz}
%\usepackage[english]{babel}
\usepackage{csquotes}
\MakeOuterQuote{"}
\usepackage{amssymb}
\usepackage{wasysym}
%\DisableQuotes

\definecolor{skyblue}{RGB}{150, 200, 255}
\definecolor{darkblue}{RGB}{20,40,120} %25,47,98}
\definecolor{lightblue}{HTML}{D0D2FF}
\definecolor{lightyellow}{HTML}{FFFFAA}
\definecolor{darkblue}{HTML}{0000BB}
\definecolor{olivegreen}{HTML}{005000}
\definecolor{violet}{HTML}{6600CC}

%\newcommand{\keyword}[1]{\@keyword{#1}\index{#1}}
\newcommand{\expword}[1]{\textcolor{olivegreen}{#1}}
\newcommand{\optword}[1]{\textcolor{violet}{\bfseries #1}}


%===============================================================
% Constants
%===============================================================

\def\LEFTIND{18pt}


%===============================================================
% Internal Varivables
%===============================================================

\def\@title{}
\def\@version{}
\def\@syear{}
\def\@sauthor{}

%===============================================================
% Helper commands (formatting)
%===============================================================

% Format an expert (First-name Family-name (email))
\newcommand{\formatexpert}[3]{\titlecap{#1}\ \titlecap{#2} (\url{#3})} %\MakeUppercase

%===============================================================
% Setters (setting internal variables)
%===============================================================

\def\settitle#1{\def\@title{#1}}
\def\setversion#1{\def\@version{#1}}
\def\setyear#1{\def\@syear{#1}}
\def\setauthor#1{\def\@sauthor{#1}}

%===============================================================
% Adding elements
%===============================================================


%===============================================================
% Headers and Footers
%===============================================================

\fancypagestyle{fancymain}{
	\fancyhf{}
	\fancyfoot[C]{
		\begin{tblr}{
				colspec = {XlXlp{1cm}},
				colsep=3pt,
				rows = {valign=m, halign=c, abovesep=6pt, belowsep=6pt},
				cell{1}{1,3}={skyblue},
				cell{1}{5}={darkblue, fg=white, font=\bfseries},
			}
			\@syear/ESI/2CSSID/NLP/\@version && \@sauthor && \thepage\\
		\end{tblr}
	}
}
\renewcommand{\headrulewidth}{0pt}

\usepackage[left=1.5cm,right=1.5cm,top=1cm,bottom=1.5cm,includefoot]{geometry}

%\ifinfo
%\usepackage[left=1.5cm,right=1.5cm,top=1cm,bottom=1cm,includefoot,includehead,headheight=1.25cm,asymmetric]{geometry}
%\pagestyle{fancyinfo}
%\else
%\ifsmp
%\usepackage[left=1.5cm,right=1.5cm,top=1.5cm,bottom=1.5cm,includefoot,includehead,headheight=14pt,asymmetric]{geometry}
%\pagestyle{fancymain}
%\thispagestyle{fancyfirstpage}
%\else
%\usepackage[left=1.5cm,right=1.5cm,top=1.5cm,bottom=1cm,includefoot,asymmetric]{geometry}
%\pagestyle{empty}
%\fi
%\fi

%===============================================================
% Printing commands (print results)
%===============================================================


%===============================================================
% Formatting commands
%===============================================================

%\AtBeginDocument{
%	\begin{center}
%		\huge\bfseries\color{darkblue} SYLLABUS
%	\end{center}
%}
%
%\AtEndDocument{
%	
%}

\newcommand{\bgparbox}[1]{\colorbox{darkblue}{\setlength{\fboxsep}{6pt}%
		
		\begin{tabularx}{\dimexpr\linewidth-\fboxsep}{l>{\raggedright\arraybackslash}X}#1\end{tabularx}}}

\titleformat{\section}[block]
{}{}{0em}
{\bgparbox{\color{white}\bfseries\Large\thesection.\hskip8pt #1}}

\titleformat{\subsection}[block]
{}{}{0em}
{{\color{darkblue}\bfseries\large\thesubsection\hskip8pt #1}}

\setlength{\parskip}{3pt}
\setlength\parindent{0pt}


%\renewcommand{\thesection}{\arabic{section}.}
\renewcommand{\thesubsection}{\arabic{section}.\arabic{subsection}.}
\renewcommand{\thesubsubsection}{\thesubsection\arabic{subsubsection}.}

\renewcommand{\baselinestretch}{1.15} 


\setlist{nolistsep}
%
%\setlist[itemize]{left=\LEFTIND}

\let\oldurl\url
\renewcommand{\url}[1]{{\footnotesize\oldurl{#1}}}


%===============================================================
% Custom Tabls
%===============================================================





%===============================================================
% Automatic calls
%===============================================================

\def\maketitle{
	\clearpage
	\thispagestyle{empty}
	
	\noindent
	\begin{tblr}{
			colspec = {llX},
			rows = {valign=m, halign=l},
			rowsep=1pt,
			row{5} = {darkblue, fg=white, font=\bfseries\LARGE, abovesep=5mm, belowsep=5mm},
			row{6} = {white, fg=darkblue, halign=r},
		}
		\SetCell[r=4]{l} \includegraphics[width=2cm]{../../../extra/logo/esi.nlp.pdf} && \'Ecole national Supérieure d'Informatique, Algiers, Algeria\\
		&& 2CSSID (\@syear)\\
		&& Natural Language Processing (NLP)\\
		&&\\
%		&&\\
		\SetCell[c=3]{c}\@version: \@title &&\\
		 && \large\bfseries\itshape\@sauthor\\
	\end{tblr}
}

\pagestyle{fancymain} 

