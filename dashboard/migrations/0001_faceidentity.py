from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="FaceIdentity",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "person_id",
                    models.CharField(db_index=True, max_length=32, unique=True),
                ),
                ("embedding", models.JSONField(default=list)),
                ("image_path", models.CharField(blank=True, max_length=500)),
                ("first_seen", models.DateTimeField(auto_now_add=True)),
                ("last_seen", models.DateTimeField(auto_now=True)),
                ("seen_count", models.PositiveIntegerField(default=1)),
                ("is_masked", models.BooleanField(default=False)),
                ("is_active", models.BooleanField(default=True)),
            ],
            options={
                "verbose_name_plural": "face identities",
                "ordering": ("-last_seen",),
            },
        ),
    ]
