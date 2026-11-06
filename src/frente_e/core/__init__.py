"""Núcleo quantitativo: funções puras.

Invariantes (verificadas em tests/unit/test_architecture.py):
- não importa as camadas data, experiments, reporting nem módulos de I/O;
- não usa o estado aleatório global do NumPy: funções estocásticas recebem um
  ``numpy.random.Generator``.
"""
