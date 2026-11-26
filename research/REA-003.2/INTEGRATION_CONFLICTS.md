# Integração

Os dois cherry-picks diretos foram aplicados com `-x` sem conflito textual. O restante foi integrado em commits pequenos, com referência ao SHA original. O guard de fila reaproveita a fila de duas prioridades e evita introduzir o book de PSOs, uma terceira fila ou workers adicionais. O backport da biblioteca usa o lock local de salvamento e restaura dirty flags, pois a correção que motivou a mudança upstream protege apenas o ajudante de escrita de texto. O parser de diagnóstico usa o leitor DXBC mantido no projeto.

A correção direta Cascade Wave32 ficou atrás do gate de execução. As otimizações de heap, journal, descritores e aliases interagem com invariantes locais de autoridade, fences e geração de recursos. Elas não integram a release P0. Não houve merge de um fork inteiro na branch candidata. A worktree separada de referência preserva o HEAD exato do Vice-Tay.
