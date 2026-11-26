# Limitações conhecidas

- A fila antecipada possui limite de entradas e concorrência. O limite não mede a RAM por shader, não cancela o compilador do driver e não elimina o descarte de draws do mecanismo existente após as esperas. Um único PSO pode consumir muita RAM ou reter um worker por longo tempo.
- A matriz de candidatos documenta riscos de porta e equivalência. Os limiares de textura, buffers e journal de GPUs AMD ou de uma RTX 4060 Ti não foram validados na RTX 2060 SUPER.
- A escolha de presets e o uso de uma cópia do estado não recriam a cronologia dos logs mais antigos. O cache interno NVIDIA está fora do isolamento. O rótulo frio se refere aos caches Victory.
- A armadilha de escopo do cache de tradução pode gerar misses mesmo quando a biblioteca D3D12 é aceita. Um resultado de integridade DXC ou D3D12 sintético não comprova renderização correta do título.
- FPS de menu, contadores cumulativos e janelas curtas de cutscene não preenchem os gates de mundo 3D, integridade visual ou desempenho histórico de gameplay.
- A matriz CSV é um artefato de pesquisa solicitado. A política de assets existente do repositório não permite sua extensão para staging. O CSV fica ao lado dos demais relatórios na worktree.
- Cada execução deve escrever em seu próprio cache. O lock de publicação é interno ao processo e uma biblioteca compartilhada entre duas instâncias continuaria suscetível a conflitos entre publicações.
- O código antigo de espera continua sendo relevante para a progressão da cena. Os parâmetros novos de escalonamento e pressão são hipóteses de teste para esta máquina.
