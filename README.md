# Final-Project

## Network Optimisation

First we create the basic framework on which we can work. We require the use of classes and inheritance to attain the maximal generality in order to fully investigate our problem.

Network architecture:

---
title: "Latex and TikZ in knitr chunks"
author: "Me"
header-includes:
   - \usepackage{xspace}
   - \usepackage{tikz}
   - \usetikzlibrary{shapes.geometric,arrows}
   - \def\TikZ{Ti\emph{k}Z\ }
output:
  pdf_document 
---

\begin{tikzpicture}[node distance=2cm, auto]  
    \tikzstyle{decision}=[diamond, draw, fill=blue!20, text width=4.5em, text badly centered, node distance=3cm, inner sep-0pt]  
    \tikzstyle{block}=[rectangle, draw, fill=blue!20, text width=5em, text badly centered, rounded corners, minimum height=4em]  
    \tikzstyle{line}=[draw, -latex']  
    \tikzstyle{terminator} = [ draw, ellipse, fill=red!20, node distance=3cm, minimum height=2em]    
    \node [terminator] (puc) {Power-Up Reset};  
    \node [block, below of=puc] (wdt)  {Stop Watchdog};  
    \node [block, below of=wdt] (port) {Setup Port Pins};  
    \node [block, below of=port] (loop) {Loop Forever};  
    \path [line] (puc)  -- (wdt);  
    \path [line] (wdt)  -- (port);  
    \path [line] (port) -- (loop);  
    \path [line] (loop) -- (loop);  
\end{tikzpicture}