"""Contrats des réponses brutes de l'API PMU turfinfo.

L'API n'est pas documentée publiquement. Ce contrat a été confronté aux vraies
réponses le 2026-10-30 (programme, participants et rapports du 29/10/2026, trot
attelé, trot monté, obstacle et plat). Les tests marqués `network`
(tests/test_collectors.py) le revalident à la demande.

Les modèles sont tolérants (`extra="ignore"`) : un champ inconnu n'est pas une erreur,
mais un champ structurant manquant l'est (ContractError côté connecteur).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Tolerant(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class PmuHippodrome(_Tolerant):
    code: str | None = None
    libelle_court: str | None = Field(default=None, alias="libelleCourt")
    libelle_long: str | None = Field(default=None, alias="libelleLong")


class PmuPays(_Tolerant):
    code: str | None = None


class PmuPenetrometre(_Tolerant):
    valeur_mesure: str | None = Field(default=None, alias="valeurMesure")
    intitule: str | None = None


class PmuCourse(_Tolerant):
    num_reunion: int = Field(alias="numReunion")
    num_ordre: int = Field(alias="numOrdre")
    libelle: str | None = None
    heure_depart: int | None = Field(default=None, alias="heureDepart")  # epoch ms
    distance: int | None = None
    distance_unit: str | None = Field(default=None, alias="distanceUnit")
    discipline: str | None = None
    specialite: str | None = None
    corde: str | None = None
    montant_prix: float | None = Field(default=None, alias="montantPrix")
    nombre_declares_partants: int | None = Field(default=None, alias="nombreDeclaresPartants")
    condition_sexe: str | None = Field(default=None, alias="conditionSexe")
    conditions: str | None = None
    statut: str | None = None
    arrivee_definitive: bool | None = Field(default=None, alias="arriveeDefinitive")
    ordre_arrivee: list[list[int]] | None = Field(default=None, alias="ordreArrivee")
    penetrometre: PmuPenetrometre | None = None


class PmuReunion(_Tolerant):
    num_officiel: int = Field(alias="numOfficiel")
    hippodrome: PmuHippodrome
    pays: PmuPays | None = None
    courses: list[PmuCourse] = Field(default_factory=list)


class PmuProgrammeBody(_Tolerant):
    date: int | None = None
    reunions: list[PmuReunion] = Field(default_factory=list)


class PmuProgramme(_Tolerant):
    programme: PmuProgrammeBody


class PmuRapportCote(_Tolerant):
    type_pari: str | None = Field(default=None, alias="typePari")
    rapport: float | None = None
    type_rapport: str | None = Field(default=None, alias="typeRapport")
    date_rapport: int | None = Field(default=None, alias="dateRapport")  # epoch ms


class PmuGains(_Tolerant):
    gains_carriere: float | None = Field(default=None, alias="gainsCarriere")


class PmuParticipant(_Tolerant):
    num_pmu: int = Field(alias="numPmu")
    nom: str
    age: int | None = None
    sexe: str | None = None
    race: str | None = None
    statut: str | None = None
    place_corde: int | None = Field(default=None, alias="placeCorde")
    driver: str | None = None
    jockey: str | None = None
    entraineur: str | None = None
    proprietaire: str | None = None
    musique: str | None = None
    nombre_courses: int | None = Field(default=None, alias="nombreCourses")
    nombre_victoires: int | None = Field(default=None, alias="nombreVictoires")
    nombre_places: int | None = Field(default=None, alias="nombrePlaces")
    gains_participant: PmuGains | None = Field(default=None, alias="gainsParticipant")
    handicap_poids: float | None = Field(default=None, alias="handicapPoids")
    poids_condition_monte: float | None = Field(default=None, alias="poidsConditionMonte")
    handicap_distance: int | None = Field(default=None, alias="handicapDistance")
    oeilleres: str | None = None
    deferre: str | None = None
    ordre_arrivee: int | None = Field(default=None, alias="ordreArrivee")
    incident: str | None = None
    temps_obtenu: int | None = Field(default=None, alias="tempsObtenu")
    reduction_kilometrique: int | None = Field(default=None, alias="reductionKilometrique")
    dernier_rapport_direct: PmuRapportCote | None = Field(
        default=None, alias="dernierRapportDirect")
    dernier_rapport_reference: PmuRapportCote | None = Field(
        default=None, alias="dernierRapportReference")


class PmuParticipants(_Tolerant):
    participants: list[PmuParticipant]


class PmuRapportLigne(_Tolerant):
    combinaison: str | None = None
    dividende_pour_un_euro: float | None = Field(default=None, alias="dividendePourUnEuro")


class PmuRapportPari(_Tolerant):
    type_pari: str = Field(alias="typePari")
    rapports: list[PmuRapportLigne] = Field(default_factory=list)
