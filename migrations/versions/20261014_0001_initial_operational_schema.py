"""Create PickFoot deployment tables."""

from alembic import op

from pickfoot.api.models import Base

revision = "20261014_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
