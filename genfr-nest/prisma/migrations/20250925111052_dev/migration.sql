-- CreateTable
CREATE TABLE "public"."EmailAction" (
    "id" SERIAL NOT NULL,
    "userEmail" TEXT NOT NULL,
    "action" TEXT NOT NULL,
    "lastSentAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "EmailAction_pkey" PRIMARY KEY ("id")
);

-- AddForeignKey
ALTER TABLE "public"."EmailAction" ADD CONSTRAINT "EmailAction_userEmail_fkey" FOREIGN KEY ("userEmail") REFERENCES "public"."User"("email") ON DELETE RESTRICT ON UPDATE CASCADE;
