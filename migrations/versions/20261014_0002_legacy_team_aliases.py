"""Persist team aliases imported from the legacy local SQLite database."""

from alembic import op

from pickfoot.api.models import TeamAlias

revision = "20261014_0002"
down_revision = "20261014_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    TeamAlias.__table__.create(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    TeamAlias.__table__.drop(bind=op.get_bind(), checkfirst=True)
