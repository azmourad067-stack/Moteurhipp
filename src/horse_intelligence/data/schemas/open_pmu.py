"""Contrat de l'API communautaire Open PMU (https://github.com/open-pmu-api/open-pmu-api).

Format documenté dans le README du projet et dans `types.ts` (audit du 2026-10-30).
Limites établies par l'audit (docs/00_audit_et_architecture.md §2.5) :
- environ une course par jour (vraisemblablement la course Quinté+), pas le programme complet ;
- `arrivee_details` ne décrit que les chevaux classés à l'arrivée, pas tout le champ ;
- `cotes` : trois valeurs sans horodatage ni définition → inutilisables pour une décision ;
- `heure_depart` en heure locale sans fuseau.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Tolerant(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class OpenPmuHorse(_Tolerant):
    nom_cheval: str | None = None
    position: int | None = None


class OpenPmuRace(_Tolerant):
    type: str | None = None
    distance: int | None = None
    prix: str | None = None
    lieu: str | None = None
    heure_depart: str | None = None
    partants: int | None = None
    non_partants: list[int] | None = None
    arrivee: list[int] | None = None
    reunion_course: str | None = Field(default=None, alias="r/c")
    date: str | None = None
    arrivee_details: dict[str, OpenPmuHorse] = Field(default_factory=dict)

    @field_validator("non_partants", mode="before")
    @classmethod
    def _non_partants(cls, v: object) -> object:
        # les anciennes données utilisent un entier (0) au lieu d'une liste
        return [] if isinstance(v, int) else v

    @property
    def reunion_course_nums(self) -> tuple[int, int] | None:
        raw = (self.reunion_course or "").upper().replace(" ", "")
        if not raw.startswith("R") or "/C" not in raw:
            return None
        r, c = raw[1:].split("/C", 1)
        return (int(r), int(c)) if r.isdigit() and c.isdigit() else None


class OpenPmuResponse(_Tolerant):
    error: bool
    total: int = 0
    message: list[OpenPmuRace] | str
