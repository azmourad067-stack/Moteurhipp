"""Conversion des réponses PMU en enregistrements normalisés.

Règles anti-fuite appliquées ici :
- une cote « directe » relevée après l'heure de départ est classée `post_start`
  (≈ cote finale) et ne pourra jamais servir à une décision simulée ;
- un résultat n'est retenu que s'il est observé après l'heure de départ ;
- les agrégats de carrière fournis par la source sont conservés tels quels mais
  marqués par `observed_after_start` au stockage.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from horse_intelligence.data.processors.cleaning import canonical_name, clean_text
from horse_intelligence.data.schemas.domain import (
    EntryRecord,
    OddsRecord,
    PayoutRecord,
    RaceRecord,
    ResultRecord,
)
from horse_intelligence.data.schemas.pmu import (
    PmuCourse,
    PmuParticipant,
    PmuProgramme,
    PmuRapportPari,
    PmuReunion,
)

# Unités confirmées contre les vraies réponses le 2026-10-30. Les contrôles qualité
# (validation.py) signalent toute valeur hors plage plausible en cas de changement.
PMU_UNITS = {
    "handicapPoids / poidsConditionMonte": "hectogrammes (580 → 58,0 kg)",
    "heureDepart / dateRapport": "epoch en millisecondes (UTC)",
    "tempsObtenu / reductionKilometrique": "millisecondes (72500 → 1'12\"5)",
    "gainsCarriere / dividendePourUnEuro": ("centimes (620 → 6,20 €) ; gains hors France en "
                                            "monnaie locale : non stockés"),
    "montantPrix": "euros",
    "dernierRapportReference": "cote publiée ~20 à 30 min avant le départ (horodatée)",
    "dernierRapportDirect": "dernière cote ; après la course = cote finale",
}
HECTOGRAMS_PER_KG = 10.0
CENTS_PER_EURO = 100.0
MS_PER_S = 1000.0


def normalize_bet_type(raw: str | None) -> str:
    """`E_SIMPLE_GAGNANT` (masse Internet) → `SIMPLE_GAGNANT` ; la masse est stockée à part."""
    return (raw or "SIMPLE_GAGNANT").removeprefix("E_")


def _ms_to_s(value: int | None) -> float | None:
    return value / MS_PER_S if value else None


def _hg_to_kg(value: float | None) -> float | None:
    return value / HECTOGRAMS_PER_KG if value else None

_DISCIPLINES = {
    "TROT_ATTELE": "trot_attele",
    "ATTELE": "trot_attele",
    "TROT_MONTE": "trot_monte",
    "MONTE": "trot_monte",
    "PLAT": "plat",
    "OBSTACLE": "obstacle",
    "HAIE": "obstacle",
    "STEEPLECHASE": "obstacle",
    "STEEPLE_CHASE": "obstacle",
    "CROSS": "obstacle",
}

FINISHED_STATUSES = {"FIN_COURSE", "ARRIVEE_DEFINITIVE", "ARRIVEE_DEFINITIVE_COMPLETE"}


def epoch_ms_to_utc(value: int | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC)


def normalize_discipline(specialite: str | None, discipline: str | None) -> str | None:
    for raw in (specialite, discipline):
        if raw and raw.upper() in _DISCIPLINES:
            return _DISCIPLINES[raw.upper()]
    return None


def race_from_pmu(day: date, reunion: PmuReunion, course: PmuCourse) -> RaceRecord:
    hippo = reunion.hippodrome
    code = hippo.code or canonical_name(hippo.libelle_court) or f"R{reunion.num_officiel}"
    status = course.statut
    return RaceRecord(
        race_date=day,
        reunion_num=reunion.num_officiel,
        course_num=course.num_ordre,
        racecourse_code=code,
        racecourse_name=clean_text(hippo.libelle_long or hippo.libelle_court),
        country=reunion.pays.code if reunion.pays else None,
        label=clean_text(course.libelle),
        discipline=normalize_discipline(course.specialite, course.discipline),
        discipline_raw=course.discipline,
        specialite_raw=course.specialite,
        distance_m=course.distance if (course.distance_unit or "METRE") == "METRE" else None,
        corde=course.corde,
        prize_eur=course.montant_prix,
        declared_runners=course.nombre_declares_partants,
        sex_condition=course.condition_sexe,
        conditions_text=clean_text(course.conditions),
        scheduled_start_utc=epoch_ms_to_utc(course.heure_depart),
        status=status,
        result_final=bool(course.arrivee_definitive) or (status or "") in FINISHED_STATUSES,
        going_label=course.penetrometre.intitule if course.penetrometre else None,
        penetrometer=course.penetrometre.valeur_mesure if course.penetrometre else None,
    )


def races_from_programme(day: date, programme: PmuProgramme) -> list[tuple[PmuReunion, RaceRecord]]:
    out = []
    for reunion in programme.programme.reunions:
        for course in reunion.courses:
            out.append((reunion, race_from_pmu(day, reunion, course)))
    return out


def entry_from_pmu(p: PmuParticipant, race_date: date,
                   earnings_in_euros: bool = True) -> EntryRecord:
    name = clean_text(p.nom) or p.nom
    return EntryRecord(
        num_pmu=p.num_pmu,
        horse_name=name,
        horse_canonical=canonical_name(name) or name,
        birth_year=(race_date.year - p.age) if p.age is not None else None,
        breed=clean_text(p.race),
        age=p.age,
        sex=p.sexe,
        rider_name=clean_text(p.driver or p.jockey),
        trainer_name=clean_text(p.entraineur),
        owner_name=clean_text(p.proprietaire),
        draw=p.place_corde,
        weight_kg=_hg_to_kg(p.handicap_poids),
        weight_carried_kg=_hg_to_kg(p.poids_condition_monte) or _hg_to_kg(p.handicap_poids),
        handicap_distance_m=p.handicap_distance,
        blinkers=p.oeilleres,
        shoes=p.deferre,
        status=p.statut,
        form_string=clean_text(p.musique),
        career_starts=p.nombre_courses,
        career_wins=p.nombre_victoires,
        career_places=p.nombre_places,
        # Constat du 2026-10-30 : gains des chevaux chiliens en monnaie locale (jusqu'à
        # ~95 M « € »). Hors France, la devise n'est pas garantie : valeur non stockée.
        career_earnings=(p.gains_participant.gains_carriere / CENTS_PER_EURO
                         if earnings_in_euros and p.gains_participant
                         and p.gains_participant.gains_carriere is not None
                         else None),
    )


def odds_from_pmu(p: PmuParticipant, observed_after_start: bool, pool: str = "OFFLINE",
                  payload_id: int | None = None) -> list[OddsRecord]:
    out: list[OddsRecord] = []
    ref = p.dernier_rapport_reference
    if ref is not None and ref.rapport:
        out.append(OddsRecord(p.num_pmu, normalize_bet_type(ref.type_pari), "reference",
                              ref.rapport, epoch_ms_to_utc(ref.date_rapport), pool, payload_id))
    direct = p.dernier_rapport_direct
    if direct is not None and direct.rapport:
        kind = "post_start" if observed_after_start else "live"
        out.append(OddsRecord(p.num_pmu, normalize_bet_type(direct.type_pari), kind,
                              direct.rapport, epoch_ms_to_utc(direct.date_rapport), pool,
                              payload_id))
    return out


def result_from_pmu(p: PmuParticipant) -> ResultRecord | None:
    if (p.statut or "").upper() == "NON_PARTANT":
        return None
    if p.ordre_arrivee is None and p.incident is None:
        return None
    return ResultRecord(
        num_pmu=p.num_pmu,
        finish_position=p.ordre_arrivee if (p.ordre_arrivee or 0) > 0 else None,
        incident=p.incident,
        time_s=_ms_to_s(p.temps_obtenu),
        reduction_km_s=_ms_to_s(p.reduction_kilometrique),
    )


def payouts_from_pmu(rapports: list[PmuRapportPari], pool: str = "OFFLINE",
                     payload_id: int | None = None) -> list[PayoutRecord]:
    return [
        PayoutRecord(normalize_bet_type(r.type_pari), clean_text(line.combinaison),
                     line.dividende_pour_un_euro / CENTS_PER_EURO
                     if line.dividende_pour_un_euro is not None else None,
                     pool, payload_id)
        for r in rapports for line in r.rapports
    ]
